import torch
from proofreader.utils.vis import *
import numpy as np
from proofreader.data.cremi import prepare_cremi_vols
from proofreader.run.log import plot_voi_curve
from torch.utils.tensorboard import SummaryWriter
import os
from collections import defaultdi


def test_baseline(num_slices, vols):
    """ 
    Here we test baseline method using the same datasets
    the canidates batches in each dataset are sorted by avg distance to top neruite
    so we can simulate the method by predicting True for first canidatate in every batch
    """

    # load dataset
    path = f'/mnt/home/jberman/ceph/pf/dataset/DATASET_ns={num_slices}_cs=2_sc=None_test.pt'
    X, Y, I = torch.load(f'{path}')
    print(len(X), len(Y), len(I))
    # assemble args
    thresholds = [0.9]
    writer = SummaryWriter(log_dir=os.path.join(
        f'/mnt/home/jberman/ceph/pf/baseline/ns_{num_slices}', 'log/baseline'))
    epoch = 1

    dglobal = defaultdict(list)

    # build info and y_hats doing merge first method
    infos = []
    y_hats = np.zeros((len(I), 2))
    for index, info_batch in enumerate(I):
        infos.append(info_batch[0])
        y_hats[index] = np.array([0.0, 1.0])
    infos = np.array(infos)

    # do voi
    plot_voi_curve(vols, infos, y_hats, thresholds,
                   num_slices, writer, dglobal, epoch)

    dglobal = json.loads(json.dumps(dglobal))
    np.save(
        f'/mnt/home/jberman/ceph/pf/baseline/ns_{num_slices}/data/baseline.npy', dglobal)


if __name__ == '__main__':

    print('reading vols...')
    train_vols, test_vols = prepare_cremi_vols(
        '/mnt/home/jberman/sc/proofreader/dataset/cremi')

    num_slices = 0
    print(f'doing baseline for ns={num_slices}...')
    test_baseline(num_slices, test_vols)
