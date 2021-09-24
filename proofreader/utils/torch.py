import torch
import gc
import os
from .all import readable_bytes

def get_all_live_tensors():
    tensors = []
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                tensors.append(obj)
        except:
            pass

    return tensors

def print_all_live_tensors():
    print('===========================TENSORS===========================')
    tensors = get_all_live_tensors()
    for t in tensors:
        print(obj.dtype, obj.device, type(obj), obj.size())



def count_all_live_tensors():
    count = 0
    mem = 0
    tensors = get_all_live_tensors()
    for t in tensors:
        count += 1
        mem += t.element_size() * t.nelement()
 
    return count, readable_bytes(mem)


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
