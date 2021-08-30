
import math
import torch
import gc
import numpy as np


def split_int(i, bias='left'):
    f = i/2
    big = math.ceil(f)
    sm = math.floor(f)
    if bias == 'left':
        return (big, sm)
    elif bias == 'right':
        return (sm, big)


def pad_2_divisible_by(vol, factor):
    vol_shape = vol.shape
    pad_width = []
    for i in range(len(factor)):
        left = vol_shape[i] % factor[i]
        if left > 0:
            add = factor[i] - left
            pad_width.append(split_int(add))
        else:
            pad_width.append((0, 0))
    padded = np.pad(vol, pad_width)
    vol_shape = padded.shape
    # check
    assert vol_shape[-1] % factor[-1] == 0 and vol_shape[-2] % factor[-2] == 0 and vol_shape[-3] % factor[-3] == 0, 'Image dimensions must be divisible by the patch size.'
    return padded


def print_all_live_tensors():
    print('===========================TENSORS===========================')
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                print(obj.dtype, obj.device, type(obj), obj.size())
        except:
            pass


def readable_bytes(nbytes):
    suffixes = ['B', 'KB', 'MB', 'GB', 'TB', 'PB']
    i = 0
    while nbytes >= 1024 and i < len(suffixes)-1:
        nbytes /= 1024.
        i += 1
    f = ('%.2f' % nbytes).rstrip('0').rstrip('.')
    return '%s %s' % (f, suffixes[i])
