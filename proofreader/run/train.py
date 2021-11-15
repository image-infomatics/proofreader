import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from proofreader.model.classifier import *
from proofreader.model.config import *
from proofreader.utils.torch import *
from proofreader.run.log import *
from proofreader.data.cremi import prepare_cremi_vols

import numpy as np
import click
from tqdm import tqdm
import os
import random
import shutil
from collections import defaultdict
import json


@click.command()
@click.option('--config',
              type=str,
              help='name of the configuration defined in the config list'
              )
@click.option('--overwrite',
              type=bool, default=False,
              help='wheather to overwrite previous run of config'
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
              type=int, default=24,
              help='number of epochs to train for'
              )
@click.option('--batch-size', '-b',
              type=int, default=800,
              help='size of mini-batch.'
              )
@click.option('--num_workers', '-w',
              type=int, default=-1,
              help='num workers for pytorch dataloader. -1 means automatically set.'
              )
@click.option('--training-interval', '-t',
              type=int, default=4, help='num times to log per epoch.'
              )
@click.option('--validation-interval', '-v',
              type=int, default=1, help='validation interval in terms of epochs to record validation data.'
              )
@click.option('--test-interval', '-s',
              type=int, default=1, help='interval when to run full test.'
              )
@click.option('--test',
              type=bool, default=False, help='whether to just run inference.'
              )
@click.option('--load',
              type=str, default=None, help='load from checkpoint, pass path to ckpt file'
              )
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


def train(config: str, overwrite: bool, path: str, seed: int, output_dir: str, epochs: int, batch_size: int, num_workers: int,
          training_interval: int, validation_interval: int, test_interval: int, test: bool,
          load: str, ddp: bool, rank: int = 0, world_size: int = 1):

    print('\nstarting...')
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

    if test:
        output_dir = f'{output_dir}_INFER'
    # global datarecorder to save outside of tensorboard
    # ''metric' -> [ (x_epoch1, y_epoch1), (x_epoch2, y_epoch2), ...]
    dglobal_valid = defaultdict(list)
    dglobal_test = defaultdict(list)

    t_writer, v_writer, s_writer = None, None, None
    if rank == 0:
        # rm dir if exists then create
        if not overwrite:
            while os.path.exists(f'{output_dir}_{version}'):
                version += 1

        output_dir = f'{output_dir}_{version}'
        if overwrite and os.path.exists(output_dir):
            shutil.rmtree(output_dir)

        os.makedirs(output_dir)

        # log config
        config_summary = config.toString()
        train_summary = f'TRAINING\nseed: {seed}, batch_size: {batch_size}, use_gpu: {use_gpu}, total_cpus: {cpus}, total_gpus: {gpus}, ddp: {ddp}, world_size: {world_size}, num_workers: {num_workers}\n'

        # clear in case was stopped before
        tqdm._instances.clear()
        t_writer = SummaryWriter(log_dir=os.path.join(output_dir, 'log/train'))
        v_writer = SummaryWriter(log_dir=os.path.join(output_dir, 'log/valid'))
        s_writer = SummaryWriter(log_dir=os.path.join(output_dir, 'log/test'))

        t_writer.add_text('config', config_summary)
        t_writer.add_text('train', train_summary)

    # data
    print('building dataset...')
    if config.dataset.path is not None:
        train_dataset, val_dataset, test_dataset = load_dataset_from_disk(
            config.dataset, config.augmentor, test=test)

    # get vols for voi
    train_vols, test_vols = prepare_cremi_vols(
        '/mnt/home/jberman/sc/proofreader/dataset/cremi')

    # model
    print('building model...')
    model, loss_module, optimizer, scheduler = build_full_model_from_config(
        config.model, config.dataset, epochs)

    # handle GPU and parallelism
    pin_memory = False
    train_sampler, val_sampler, test_sampler = None, None, None
    if config.dataset.balance_samples:
        train_sampler = BalanceClassSampler(
            list(train_dataset.y[:, 0].numpy()), mode='upsampling')

    if use_gpu:
        # gpu with DistributedDataParallel
        if ddp:
            model = model.cuda(rank)
            model = DistributedDataParallel(
                model, device_ids=[rank])
            train_sampler = DistributedSampler(
                train_dataset, world_size, rank, seed=seed, shuffle=True)
            val_sampler = DistributedSampler(
                val_dataset, world_size, rank, seed=seed, shuffle=True)
            test_sampler = DistributedSampler(
                test_dataset, world_size, rank, seed=seed, shuffle=True)
        # gpu with DataParallel
        else:
            model = model.cuda()
            model = nn.DataParallel(model)
        # any gpu use
        pin_memory = True

    if load is not None:
        model = load_model(model, load)

    print('building dataloader...')
    val_workers = 4

    def collate_info(b):
        xs, ys, infos = [[t[i] for t in b] for i in range(len(b[0]))]
        return torch.stack(xs), torch.stack(ys), np.array(infos)

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                                  num_workers=num_workers-val_workers, pin_memory=pin_memory, sampler=train_sampler, drop_last=True, shuffle=(train_sampler is None))
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size,
                                num_workers=val_workers, pin_memory=pin_memory, sampler=val_sampler, drop_last=True, shuffle=(val_sampler is None), collate_fn=collate_info)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size,
                                 num_workers=val_workers, pin_memory=pin_memory, sampler=test_sampler, drop_last=True, shuffle=(test_sampler is None), collate_fn=collate_info)

    total_train_batches = len(train_dataloader)

    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    print("starting training...")
    if rank == 0:
        t_writer.add_text('train_dataset', str(train_dataset.stats))
        t_writer.add_text('val_dataset',  str(val_dataset.stats))
        t_writer.add_text('test_dataset',  str(test_dataset.stats))
        pbar = tqdm(total=total_train_batches)

    for epoch in range(epochs):
        if rank == 0:
            pbar.reset(total=total_train_batches)
            pbar.set_description(f'Training {epoch}')
        if ddp:
            train_sampler.set_epoch(epoch)
            val_sampler.set_epoch(epoch)
            test_sampler.set_epoch(epoch)

        accumulated_loss = 0.0
        all_acc = {}
        # TRAIN EPOCH
        if not test:
            for step, batch in enumerate(train_dataloader):

                # get batch
                x, y = batch

                optimizer.zero_grad(set_to_none=True)

                bid = y[..., 1]
                y = y[..., 0]

                if use_gpu:
                    x = x.cuda(rank, non_blocking=True)
                    y = y.cuda(rank, non_blocking=True)

                # foward pass
                y_hat = model(x)

                # compute loss
                loss = loss_module(y_hat, y)

                loss.backward()
                optimizer.step()
                scheduler.step()

                cur_loss = loss.item()
                accumulated_loss += cur_loss
                for t in thresholds:
                    pred = predict_class(y_hat, true_threshold=t)
                    acc = count_succ_and_errs(y, pred)
                    all_acc[t] = {k: acc.get(k, 0) + all_acc.get(t, {}).get(k, 0)
                                  for k in set(acc)}

                # record progress / metrics
                if rank == 0:
                    pbar.update(1)

            if rank == 0:
                per_example_loss = accumulated_loss / (total_train_batches)
                t_writer.add_scalar('Loss', per_example_loss, epoch)
                plot_merge_curve(all_acc, thresholds, t_writer,
                                 train_dataset.stats, 'merge_curve', None, epoch)

                accumulated_loss = 0.0
                all_acc = {}

        # VAL STEP
        if validation_interval != 0 and (epoch+1) % validation_interval == 0:
            for cur_dataloader, dataset, writer, dglobal, run in zip([val_dataloader, test_dataloader], [val_dataset, test_dataset], [v_writer, s_writer], [dglobal_valid, dglobal_test], ['Val', 'Test']):
                accumulated_loss = 0.0
                all_acc = {}
                y_hats, ys, bids, infos = [], [], [], []
                mesh_err, mesh_miss = [], []
                if rank == 0:
                    pbar.reset(total=len(cur_dataloader))
                    pbar.set_description(f'{run} {epoch}')
                model.eval()

                with torch.no_grad():
                    for step, batch in enumerate(cur_dataloader):
                        # get batch
                        x, y, info = batch
                        bid = y[..., 1]
                        y = y[..., 0]

                        if use_gpu:
                            x = x.cuda(rank, non_blocking=True)
                            y = y.cuda(rank, non_blocking=True)

                        y_hat = model(x)

                        # compute loss
                        loss = loss_module(y_hat, y)

                        # record validation metrics
                        cur_loss = loss.item()
                        accumulated_loss += cur_loss

                        for t in thresholds:
                            pred = predict_class(y_hat, true_threshold=t)
                            if t == 0.5:
                                acc, indices = count_succ_and_errs(
                                    y, pred, return_indices=True)
                                if rank == 0:
                                    # merge when should not
                                    if acc['false_positive'] >= 1:
                                        mesh = torch.swapaxes(
                                            x[indices['false_positive']][:1], -1, -2)
                                        mesh_err.append(mesh[0])

                                    # miss the merge
                                    if acc['false_negative'] >= 1:
                                        mesh = torch.swapaxes(
                                            x[indices['false_negative']][:1], -1, -2)
                                        mesh_miss.append(mesh[0])

                            else:
                                acc = count_succ_and_errs(y, pred)
                            all_acc[t] = {k: acc.get(
                                k, 0) + all_acc.get(t, {}).get(k, 0) for k in set(acc)}

                        # for single_canidate_prediction
                        y_hats.append(torch.exp(y_hat))
                        ys.append(y)
                        bids.append(bid)
                        infos.append(info)

                        if rank == 0:
                            pbar.update(1)

                    if rank == 0:
                        # meshes
                        plot_lim = 0.1
                        if config.dataset.scale:
                            plot_lim = 0.5

                        if len(mesh_err) > 0:
                            figs = build_mesh_figs(mesh_err, lim=plot_lim)
                            writer.add_figure(
                                'false_positive', figs, global_step=epoch)
                        if len(mesh_miss) > 0:
                            figs = build_mesh_figs(mesh_miss, lim=plot_lim)
                            writer.add_figure(
                                'false_negative', figs, global_step=epoch)

                        writer.add_scalar(
                            'Loss',   accumulated_loss / (len(cur_dataloader)), epoch)

                        # flatten batches
                        ys, y_hats, bids, infos = torch.cat(
                            ys), torch.cat(y_hats), torch.cat(bids), np.concatenate(infos)

                        # max batch predication
                        max_acc = {}
                        for t in thresholds:
                            max_acc[t] = max_canidate_prediction(
                                y_hats, ys, bids, threshold=t)

                        plot_merge_curve(
                            all_acc, thresholds, writer, dataset.stats, 'merge_curve', dglobal, epoch)
                        plot_merge_curve(
                            max_acc, thresholds, writer, dataset.stats, 'merge_curve_max', dglobal, epoch)

                        # Normal PR curve
                        writer.add_pr_curve(
                            'pr_curve', ys, y_hats[:, 1], global_step=epoch, num_thresholds=len(thresholds))

                        # voi
                        if run == 'Test':
                            y_hats = y_hats.cpu().numpy()
                            plot_voi_curve(test_vols, infos,
                                           y_hats, thresholds, config.dataset.num_slices, writer, dglobal, epoch)

                        plt.close('all')

        save_model(model, output_dir, epoch=epoch,
                   optimizer=optimizer)
        model.train()

    if rank == 0:
        # removes defaultdict
        dglobal_valid = json.loads(json.dumps(dglobal_valid))
        dglobal_test = json.loads(json.dumps(dglobal_test))
        # save
        datadir = os.path.join(output_dir, 'data/')
        if not os.path.exists(datadir):
            os.makedirs(datadir)

        np.save(os.path.join(datadir, 'valid.npy'), dglobal_valid)
        np.save(os.path.join(datadir, 'test.npy'), dglobal_test)
        # close
        t_writer.close()
        v_writer.close()
        pbar.close()

    print('done!')


if __name__ == '__main__':
    train_wrapper()
