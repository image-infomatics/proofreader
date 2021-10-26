import cc3d
import os
import numpy as np
from proofreader.utils.io import from_h5
from proofreader.utils.data import zero_classes_with_zspan_less_than, zero_classes_with_min_volume


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


def clean_cremi_vols(vols):
    # we must clean the CREMI vols because they contain noise even in the GT segmentations
    for i in range(len(vols)):
        vols[i] = zero_classes_with_zspan_less_than(vols[i], 3)
        vols[i] = zero_classes_with_min_volume(vols[i], 800)
        vols[i] = cc3d.connected_components(vols[i])
    return vols


def prepare_cremi_vols(path, validation_slices=None):
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

    if validation_slices is not None:
        vs = validation_slices*-1
        val_vols = [trueA_train[vs:].copy(), trueB_train[vs:].copy(),
                    trueC_train[vs:].copy()]
        train_vols = [trueA_train[:vs], trueB_train[:vs], trueC_train[:vs]]
        # redo connected_components to reconnect neurites
        val_vols = clean_cremi_vols(val_vols)
    # redo connected_components to reconnect neurites
    train_vols = clean_cremi_vols(train_vols)
    test_vols = clean_cremi_vols(test_vols)

    if validation_slices is not None:
        return train_vols, val_vols, test_vols

    return train_vols, test_vols
