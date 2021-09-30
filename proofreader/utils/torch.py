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
        print(t.dtype, t.device, type(t), t.size())


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


def save_model(model, path, epoch=None, optimizer=None, loss=None):
    """ Save trained network as file

    Args:
        model (nn.Module): current model
        path (str): file path of saved model
        epoch (int): current epoch
        optimizer (Optimizer): current optimizer
        loss (float): current loss
    """

    path = os.path.join(path, 'checkpoints')

    # make folder if doesnt exist
    if not os.path.exists(path):
        os.makedirs(path)

    fname = os.path.join(path, f'{epoch}.ckpt')
    state = {'epoch': epoch,
             'model_state_dict': model.state_dict(),
             'optimizer_state_dict': optimizer.state_dict(),
             'loss': loss, }
    torch.save(state, fname)


def load_model(model, path, optimizer=None, map_location=None, strict=True):
    checkpoint = torch.load(path, map_location=map_location)

    model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return model, optimizer
    return model


class MultiEpochsDataLoader(torch.utils.data.DataLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """ Sampler that repeats forever.
    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)
