import cc3d
import os
import numpy as np
from proofreader.utils.io import from_h5


def read_cremi_volume(volname: str, img: bool = False, seg: bool = False, pad: bool = False, path: str = './dataset'):

    assert img or seg

    filename = f'sample_{volname}'
    if pad:
        filename += '_pad'
    full_path = os.path.join(path, f'{filename}.hdf')

    if img and seg:
        i = from_h5(full_path, dataset_path='volumes/raw')
        s = from_h5(full_path, dataset_path='volumes/labels/neuron_ids')
        return (i, s)
    if img:
        return from_h5(full_path, dataset_path='volumes/raw')
    if seg:
        return from_h5(full_path, dataset_path='volumes/labels/neuron_ids')


# Clean C, some slices are totally mislabeled
def clean_true_c(vol):
    assert vol.shape[0] == 125, 'vol may have already been cleaned'
    return np.delete(vol, [14, 74], 0)

# Clean B, major slipe at 16


def clean_true_B(vol):
    assert vol.shape[0] == 125, 'vol may have already been cleaned'
    return vol[16:].copy()


def prepare_cremi_vol(path):
    trueA = read_cremi_volume('A', seg=True, path=path)
    trueB = read_cremi_volume('B', seg=True, path=path)
    trueC = read_cremi_volume('C', seg=True, path=path)
    # redo connected_components to reconnect neurites

    trueA = cc3d.connected_components(trueA)

    trueB = clean_true_B(trueB)
    trueB = cc3d.connected_components(trueB)

    trueC = clean_true_c(trueC)
    trueC = cc3d.connected_components(trueC)

    return [trueA, trueB, trueC]
