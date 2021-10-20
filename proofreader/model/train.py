from re import X
from numpy.core.numeric import zeros_like
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from proofreader.data.cremi import prepare_cremi_vols
from proofreader.model.classifier import *
from proofreader.model.config import *
from proofreader.utils.torch import *
from proofreader.model.test import test_model

import numpy as np
import click
from tqdm import tqdm
import os
import random
import shutil
import matplotlib.pyplot as plt


def build_mesh_figs(data, title='', marker='.', lim=None):
    figs = []
    for mesh in data:
        mesh = mesh.cpu()
        x, y, z = mesh[:, 0], mesh[:, 1], mesh[:, 2]
        fig = plt.figure(figsize=(4, 4))
        ax = fig.add_subplot(111, projection='3d', title=title)

        if lim is not None:
            ax.set_xlim3d(-lim, lim)
            ax.set_ylim3d(-lim, lim)
            ax.set_zlim3d(-lim, lim)

        ax.scatter(x, y, z, marker=marker)
        figs.append(fig)

    return figs


def build_curve_fig(x, y, thresholds=None, rm_last=False):

    if rm_last:
        x = x[:-1]
        y = y[:-1]
        thresholds = thresholds[:-1]

    fig = plt.figure(figsize=(8, 8))
    plt.plot(x, y, zorder=1)
    plt.scatter(x, y, c=thresholds, s=26, edgecolors='k', zorder=2)
    plt.xlabel('Error')
    plt.ylabel('Success')
    plt.title('Merge Error and Success for Thresholds')
    plt.xlim([0.0, 0.65])
    plt.ylim([0.5, 1.0])
    plt.grid()
    # label
    if thresholds is not None:
        for i in range(len(thresholds)):
            plt.annotate(f"{thresholds[i]}", (x[i], y[i]), textcoords="offset points",  xytext=(
                20, -5),  ha='center')

    return fig


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
              type=int, default=100,
              help='number of epochs to train for'
              )
@click.option('--batch-size', '-b',
              type=int, default=256,
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
          training_interval: int, validation_interval: int, test_interval: int,
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
            config.dataset, config.augmentor)
    else:
        train_vols, test_vols = prepare_cremi_vols(path)
        train_dataset, val_dataset = build_dataset_from_config(
            config.dataset, config.augmentor, train_vols)
    # model
    print('building model...')
    model, loss_module, optimizer = build_full_model_from_config(
        config.model, config.dataset)

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
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                                  num_workers=num_workers-val_workers, pin_memory=pin_memory, sampler=train_sampler, drop_last=True, shuffle=(train_sampler is None))
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size,
                                num_workers=val_workers, pin_memory=pin_memory, sampler=val_sampler, drop_last=False, shuffle=(val_sampler is None))
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size,
                                 num_workers=val_workers, pin_memory=pin_memory, sampler=test_sampler, drop_last=False, shuffle=(test_sampler is None))

    total_train_batches = len(train_dataloader)

    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    print("starting training...")
    if rank == 0:
        t_writer.add_text('train_dataset', str(train_dataset.info))
        t_writer.add_text('val_dataset',  str(val_dataset.info))
        t_writer.add_text('test_dataset',  str(test_dataset.info))
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
                             train_dataset.info, 'merge_curve', epoch)

            accumulated_loss = 0.0
            all_acc = {}

        # VAL STEP
        if validation_interval != 0 and (epoch+1) % validation_interval == 0:
            for cur_dataloader, dataset, writer in zip([val_dataloader, test_dataloader], [val_dataset, test_dataset], [v_writer, s_writer]):
                accumulated_loss = 0.0
                all_acc = {}
                y_hats, ys, bids = [], [], []
                mesh_err, mesh_miss = [], []
                if rank == 0:
                    pbar.reset(total=len(cur_dataloader))
                    pbar.set_description(f'Val {epoch}')
                model.eval()

                with torch.no_grad():
                    for step, batch in enumerate(cur_dataloader):
                        # get batch
                        x, y = batch

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
                                        # writer.add_mesh(
                                        #     'false_positive_pc', mesh, global_step=epoch)

                                    # miss the merge
                                    if acc['false_negative'] >= 1:
                                        mesh = torch.swapaxes(
                                            x[indices['false_negative']][:1], -1, -2)
                                        mesh_miss.append(mesh[0])
                                        # writer.add_mesh(
                                        #     'false_negative_pc', mesh, global_step=epoch)

                            else:
                                acc = count_succ_and_errs(y, pred)
                            all_acc[t] = {k: acc.get(
                                k, 0) + all_acc.get(t, {}).get(k, 0) for k in set(acc)}

                        # for single_canidate_prediction
                        y_hats.append(torch.exp(y_hat))
                        ys.append(y)
                        bids.append(bid)

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

                        # max batch predication
                        ys, y_hats, bids = torch.cat(
                            ys), torch.cat(y_hats), torch.cat(bids)
                        max_acc = {}
                        for t in thresholds:
                            max_acc[t] = max_canidate_prediction(
                                y_hats, ys, bids, threshold=t)

                        plot_merge_curve(
                            all_acc, thresholds, writer, dataset.info, 'merge_curve', epoch)
                        plot_merge_curve(
                            max_acc, thresholds, writer, dataset.info, 'merge_curve_max', epoch)

                        # Normal PR curve
                        writer.add_pr_curve(
                            'pr_curve', ys, y_hats[:, 1], global_step=epoch, num_thresholds=len(thresholds))

                        plt.close('all')

        save_model(model, output_dir, epoch=epoch,
                   optimizer=optimizer)
        model.train()

    if rank == 0:
        t_writer.close()
        v_writer.close()
        pbar.close()


def plot_merge_curve(accs, thresholds, writer, dataset_info, tag, epoch):

    total_neurites = dataset_info['total_neurites']
    merge_opps = dataset_info['merge_opportunities']
    # normal curve
    succ, err, tp, fp, tn, fn = [], [], [], [], [], []
    for t in thresholds:
        t_acc = accs[t]
        succ.append(
            t_acc['true_positive'] / merge_opps)
        err.append(
            t_acc['false_positive'] / total_neurites)

        tp.append(t_acc['true_positive'])
        fp.append(t_acc['false_positive'])
        tn.append(t_acc['true_negative'])
        fn.append(t_acc['false_negative'])

    # merge curve
    writer.add_pr_curve_raw(tag, tp, fp, tn, fn,
                            succ, err, global_step=epoch, num_thresholds=len(thresholds))

    # merge curve plt figures
    fig = build_curve_fig(err, succ, thresholds=thresholds, rm_last=True)
    writer.add_figure(f'{tag}_plt', fig,
                      global_step=epoch)


if __name__ == '__main__':
    train_wrapper()
