import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


from proofreader.data.cremi import prepare_cremi_vols
from proofreader.model.classifier import *
from proofreader.model.config import *
from proofreader.utils.torch import *

import numpy as np
import click
from tqdm import tqdm
import os
import random


@click.command()
@click.option('--config',
              type=str,
              help='name of the configuration defined in the config list'
              )
@click.option('--path',
              type=str, default='./dataset/cremi',
              help='path to the training data'
              )
@click.option('--seed',
              type=int, default=7,
              help='for reproducibility'
              )
@click.option('--output-dir', '-o',
              type=str, default='/mnt/home/jberman/ceph/pf',
              help='for output'
              )
@click.option('--epochs', '-e',
              type=int, default=1000,
              help='number of epochs to train for'
              )
@click.option('--batch-size', '-b',
              type=int, default=128,
              help='size of mini-batch.'
              )
@click.option('--num_workers', '-w',
              type=int, default=-1,
              help='num workers for pytorch dataloader. -1 means automatically set.'
              )
@click.option('--training-interval', '-t',
              type=int, default=200, help='training interval in terms of examples seen to record data points.'
              )
@click.option('--validation-interval', '-v',
              type=int, default=10, help='validation interval in terms of epochs to record validation data.'
              )
@click.option('--test-interval', '-ti',
              type=int, default=500, help='interval when to run full test.'
              )
@click.option('--load',
              type=str, default='', help='load from checkpoint, pass path to ckpt file'
              )
# @click.option('--use-amp',
#               type=bool, default=True, help='whether to use distrubited automatic mixed percision.'
#               )
@click.option('--ddp',
              type=bool, default=False, help='whether to use distrubited data parallel vs normal data parallel.'
              )
def train_wrapper(*args, **kwargs):
    if kwargs['ddp']:
        world_size = torch.cuda.device_count()
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        print(f'starting ddp world size {world_size}')
        mp.spawn(train_parallel, nprocs=world_size, args=(world_size, kwargs,))
    else:
        train(**kwargs)


def train_parallel(rank, world_size, kwargs):
    dist.init_process_group("nccl", init_method='env://',
                            rank=rank, world_size=world_size)

    kwargs['world_size'] = world_size
    kwargs['rank'] = rank

    train(**kwargs)


def train(config: str, path: str, seed: int, output_dir: str, epochs: int, batch_size: int, num_workers: int,
          training_interval: int, validation_interval: int, test_interval: int,
          load: str, ddp: bool, rank: int = 0, world_size: int = 1):

    # seed
    random.seed(seed)

    # config
    config_name = config
    config = get_config(config_name)

    use_gpu = torch.cuda.is_available()
    gpus = torch.cuda.device_count()
    cpus = get_cpu_count()  # gets machine cpus

    # auto set
    if num_workers == -1:
        if ddp:
            num_workers = cpus//world_size
        elif not use_gpu:
            num_workers = 1
        else:
            num_workers = cpus

    version = 0
    output_dir = f'{output_dir}/{config.name}'
    if rank == 0:
        # rm dir if exists then create
        while os.path.exists(f'{output_dir}_{version}'):
            version += 1

        output_dir = f'{output_dir}_{version}'
        os.makedirs(output_dir)

        # write config
        f = open(f"{output_dir}/config.txt", "w")
        f.write(config.toString())
        f.write(
            f'TRAINING\nseed: {seed}, batch_size: {batch_size}, use_gpu: {use_gpu}, total_cpus: {cpus}, total_gpus: {gpus}, ddp: {ddp}, world_size: {world_size}, num_workers: {num_workers}\n')
        f.close()

        # clear in case was stopped before
        tqdm._instances.clear()
        t_writer = SummaryWriter(log_dir=os.path.join(output_dir, 'log/train'))
        v_writer = SummaryWriter(log_dir=os.path.join(output_dir, 'log/valid'))

    # data
    train_vols, test_vols = prepare_cremi_vols(path)
    train_dataset, val_datset = build_dataset_from_config(
        config.dataset, config.augmentor, train_vols)

    # model
    model, loss_module, optimizer = build_full_model_from_config(
        config.model, config.dataset)

    # handle GPU and parallelism
    pin_memory = False
    train_sampler, val_sampler = None, None
    if use_gpu:
        # gpu with DistributedDataParallel
        if ddp:
            model = model.cuda(rank)
            model = DistributedDataParallel(
                model, device_ids=[rank])
            train_sampler = DistributedSampler(
                train_dataset, world_size, rank, seed=seed)
            val_sampler = DistributedSampler(
                val_datset, world_size, rank, seed=seed)
        # gpu with DataParallel
        else:
            model = model.cuda()
            model = nn.DataParallel(model)
        # any gpu use
        pin_memory = True

    val_workers = 4
    train_dataloader = MultiEpochsDataLoader(dataset=train_dataset, batch_size=batch_size,
                                             num_workers=num_workers, pin_memory=pin_memory, sampler=train_sampler, drop_last=True, persistent_workers=True)
    val_dataloader = MultiEpochsDataLoader(dataset=val_datset, batch_size=batch_size,
                                           num_workers=val_workers, pin_memory=pin_memory, sampler=val_sampler, drop_last=True)

    if rank == 0:
        print("starting...")
        pbar = tqdm(total=len(train_dataloader))

    example_number = 0
    for epoch in range(epochs):
        if rank == 0:
            pbar.refresh()
            pbar.reset()
            pbar.set_description(f'Epoch {epoch}')

        steps_since_training_interval = 0
        accumulated_loss = 0.0
        accumulated_acc = 0.0
        # TRAIN EPOCH
        for step, batch in enumerate(train_dataloader):
            example_number += batch_size
            steps_since_training_interval += 1
            # get batch
            x, y = batch

            optimizer.zero_grad(set_to_none=True)

            if use_gpu:
                x = x.cuda(rank, non_blocking=True)
                y = y.cuda(rank, non_blocking=True)

            # foward pass
            y_hat = model(x)

            # compute loss
            loss = loss_module(y_hat, y)

            loss.backward()
            optimizer.step()

            # record metrics
            cur_loss = loss.item()
            accumulated_loss += cur_loss
            pred = predict_class(y_hat)
            accs = get_accuracy(y, pred)
            accumulated_acc += accs['total_acc']

            # record progress
            if rank == 0:
                pbar.set_postfix({'cur_loss': round(cur_loss, 3)})
                pbar.update(1 * world_size)
                if example_number % training_interval == 0:

                    per_example_loss = round(
                        accumulated_loss / steps_since_training_interval, 3)
                    per_example_acc = round(
                        accumulated_acc / steps_since_training_interval, 3)

                    t_writer.add_scalar(
                        'Loss', per_example_loss, example_number)
                    t_writer.add_scalar(
                        'Accuracy', per_example_acc, example_number)
                    accumulated_loss = 0.0
                    accumulated_acc = 0.0
                    steps_since_training_interval = 0

        # VAL STEP
        if validation_interval != 0 and epoch % validation_interval == 0:
            accumulated_loss = 0.0
            accumulated_acc = 0.0
            if rank == 0:
                pbar.refresh()
                pbar.reset()
                pbar.set_description(f'Validation')
            with torch.no_grad():
                for step, batch in enumerate(val_dataloader):
                    # get batch
                    x, y = batch

                    if use_gpu:
                        x = x.cuda(rank, non_blocking=True)
                        y = y.cuda(rank, non_blocking=True)

                        y_hat = model(x)
                        # compute loss
                        loss = loss_module(y_hat, y)

                    # record metrics
                    cur_loss = loss.item()
                    accumulated_loss += cur_loss
                    pred = predict_class(y_hat)
                    accs = get_accuracy(y, pred)
                    accumulated_acc += accs['total_acc']
                    if rank == 0:
                        pbar.update(1 * world_size)

            # record validation
            if rank == 0:
                per_example_loss = round(
                    accumulated_loss / len(val_dataloader), 3)
                per_example_acc = round(
                    accumulated_acc / len(val_dataloader), 3)

                v_writer.add_scalar(
                    'Loss', per_example_loss, example_number)
                v_writer.add_scalar(
                    'Accuracy', per_example_acc, example_number)

        # VAL STEP
    if rank == 0:
        t_writer.close()
        v_writer.close()
        pbar.close()


if __name__ == '__main__':
    train_wrapper()
