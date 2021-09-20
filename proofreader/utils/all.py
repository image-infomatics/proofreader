import math
import torch
import gc
import os


def print_all_live_tensors():
    print('===========================TENSORS===========================')
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                print(obj.dtype, obj.device, type(obj), obj.size())
        except:
            pass


def get_cpu_count():
    cpu_count = None
    if hasattr(os, 'sched_getaffinity'):
        try:
            cpu_count = len(os.sched_getaffinity(0))
            return cpu_count
        except:
            pass

    cpu_count = os.cpu_count()
    if cpu_count is not None:
        return cpu_count

    try:
        import multiprocessing
        cpu_count = multiprocessing.cpu_count()
    except:
        pass

    print('could not get cpu count, returning 1')

    return 1


def readable_bytes(nbytes):
    suffixes = ['B', 'KB', 'MB', 'GB', 'TB', 'PB']
    i = 0
    while nbytes >= 1024 and i < len(suffixes)-1:
        nbytes /= 1024.
        i += 1
    f = ('%.2f' % nbytes).rstrip('0').rstrip('.')
    return '%s %s' % (f, suffixes[i])


def split_int(i, bias='left'):
    f = i/2
    big = math.ceil(f)
    sm = math.floor(f)
    if bias == 'left':
        return (big, sm)
    elif bias == 'right':
        return (sm, big)


def list_remove(arr, rm):
    if not isinstance(rm, list):
        rm = [rm]
    return [x for x in arr if not x in rm]


def clamp(n, minn, maxn):
    return max(min(maxn, n), minn)


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
