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


# Clean B, major slipe at 16


def clean_true_B(vol):
    assert vol.shape[0] == 125, 'vol may have already been cleaned'
    return vol[16:].copy()


def prepare_cremi_vols(path):
    trueA = read_cremi_volume('A', seg=True, path=path)
    trueB = read_cremi_volume('B', seg=True, path=path)
    trueC = read_cremi_volume('C', seg=True, path=path)

    # A is clean
    trueA_test = trueA[:16].copy()
    trueA_train = trueA[16:].copy()

    # Clean B, major slip at 16, we can conviently use this as test split
    assert trueB.shape[0] == 125, 'vol may have already been cleaned'
    trueB_test = trueB[:16].copy()
    trueB_train = trueB[16:].copy()

    # Clean C, some slices are totally mislabeled
    trueC = np.delete(trueC, [14, 74], 0)
    trueC_test = trueC[:16].copy()
    trueC_train = trueC[16:].copy()

    train_vols = [trueA_train, trueB_train, trueC_train]
    test_vols = [trueA_test, trueB_test, trueC_test]

    # redo connected_components to reconnect neurites
    for i in range(len(train_vols)):
        train_vols[i] = cc3d.connected_components(train_vols[i])

    for i in range(len(test_vols)):
        test_vols[i] = cc3d.connected_components(test_vols[i])

    return train_vols, test_vols
