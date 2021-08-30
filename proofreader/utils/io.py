import h5py
import os
import numpy as np


def from_h5(file_name: str,
            dataset_path: str = '/main',
            get_offset: tuple = False):

    assert os.path.exists(file_name)
    assert h5py.is_hdf5(file_name)

    with h5py.File(file_name, 'r') as f:
        arr = np.asarray(f[dataset_path])

        if get_offset:
            offset = f["/annotations"].attrs["offset"]
            # resolution is hard coded
            pixel_offset = (int(offset[0] / 40),
                            int(offset[1] / 4), int(offset[2] // 4))
            return arr, pixel_offset

    return arr


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
