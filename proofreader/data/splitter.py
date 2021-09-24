from proofreader.data.augment import Augmentor
from typing import List, Tuple
import numpy as np
import torch
import random
import cc3d
from proofreader.utils.all import list_remove, split_int
from proofreader.utils.vis import make_histogram
from proofreader.utils.data import random_sample_arr, circular_mask, crop_where
from proofreader.data.augment import Augmentor
from proofreader.data.cremi import prepare_cremi_vols
from skimage.segmentation import find_boundaries
from scipy import ndimage
import time
from torch.utils.data import DataLoader
from proofreader.utils.torch import *

def get_classes_sorted_by_volume(vol, reverse=False, return_counts=False):

    classes, counts = np.unique(vol, return_counts=True)

    sort_indices = np.argsort(counts)
    if reverse:
        sort_indices = np.flip(sort_indices)
    classes = classes[sort_indices]
    if return_counts:
        counts = counts[sort_indices]
        return classes, counts
    return classes


def get_classes_which_zspan_at_least(vol, span):
    counts = {}
    for i in range(vol.shape[0]):
        slice = vol[i]
        classes = np.unique(slice)
        for c in classes:
            if c in counts:
                counts[c] += 1
            else:
                counts[c] = 0
    res = []
    for c, cnt in counts.items():
        if cnt >= span:
            res.append(c)

    return res


def get_classes_with_at_least_volume(vol, min_volume):
    classes, counts = get_classes_sorted_by_volume(
        vol, return_counts=True, reverse=True)
    for i, cnt in enumerate(counts):
        if cnt < min_volume:
            break
    return classes[:i]


def convert_grid_to_pointcloud(vol, threshold=0, keep_features=False):
    (sz, sy, sx) = vol.shape

    # generate all coords in img
    cords = np.mgrid[0:sz, 0:sy, 0:sx]
    if keep_features:
        cords = np.append([vol], cords, axis=0)
    # select cords where above threshold
    cords = cords[:, vol > threshold]

    cords = np.swapaxes(cords, 0, 1)

    return cords.astype(np.float32)


class SplitterDataset(torch.utils.data.Dataset):
    def __init__(self,
                 vols: List,
                 num_slices: int,
                 radius: int,
                 context_slices: int,
                 num_points: int = None,
                 open_vol: bool = True,
                 retry: bool = True,
                 epoch_multplier: int = 1,
                 shuffle: bool = False,
                 verbose: bool = False,
                 return_vol: bool = False,
                 swapaxes: bool = True,
                 torch: bool = True,
                 Augmentor: Augmentor = Augmentor(),
                 ):
        """
        Parameters:
            vols (List): list of vols to use.
            num_slices (int or 2-List[a,b]): the number of slices to drop, or range of slices to drop.
            radius (int): radius (voxels) on bottom cross section in which to select second neurite.
            context_slices (int): max number of slice on top and bottom neurites.
            num_points (int): ensures number of points in pointcloud is exactly this.
            open_vol (bool): whether to get the exterior of the vol as open tube-like manifolds (True) or as closed balloon-like manifolds (False)
                             amounts to removing the interior along the z-axis (True) or for entire vol at once (False).
            retry (bool): whether to retry until success if cannot generate example.
            epoch_multplier (int): factor which scales the effective length of the dataset. Each neurite will be sampled this many times.
            shuffle (bool): whether to shuffle the classes.
            verbose (bool): print warning messages.
            return_vol (bool): whether to also return the volumetirc representation.
            swapaxes (bool): whether to swap axes (0,1). True -> CxN, False -> NxC.
            torch (bool): whether to convert to torch tensors.
            Augmentor (PCAugmentor or None): object to peform point cloud data augmentation.
        """

        super().__init__()
        print('init dataset')
        if isinstance(num_slices, int):
            num_slices = [num_slices, num_slices]
        else:
            assert len(
                num_slices) == 2, 'if num_slices is list must be len == 2 in indicating range.'

        self.vols = vols
        self.classes = []  # all possible neurite classes
        self.class_i_to_vol_i = []  # maps a neurite class index to vol index
        self.num_slices = num_slices
        self.radius = radius
        self.context_slices = context_slices
        self.num_points = num_points
        self.open_vol = open_vol
        self.retry = retry
        self.epoch_multplier = epoch_multplier
        self.shuffle = shuffle
        self.verbose = verbose
        self.return_vol = return_vol
        self.swapaxes = swapaxes
        self.torch = torch
        self.Augmentor = Augmentor

        # get classes to use
        length = 0
        for vol_i, vol in enumerate(vols):
            # TODO figure out zspan condition other than max num_slices
            classes_z = get_classes_which_zspan_at_least(vol, num_slices[1]+2)
            classes_vol = get_classes_with_at_least_volume(vol, 400)
            cls_union = list(set(classes_z) & set(classes_vol))
            length += len(cls_union)
            self.classes.extend(cls_union)
            self.class_i_to_vol_i.extend([vol_i] * len(cls_union))

        # convert to numpy
        self.classes = np.array(self.classes)
        self.class_i_to_vol_i = np.array(self.class_i_to_vol_i)

        # shuffle classes and vol_i map together
        if self.shuffle:
            shuffler = np.random.permutation(len(self.classes))
            self.classes = self.classes[shuffler]
            self.class_i_to_vol_i = self.class_i_to_vol_i[shuffler]

        # double for positive and negative examples
        self.true_length = length*2
        self.effective_length = self.true_length * self.epoch_multplier

    """
    Gets a postive or negative example from vol using some class seed. Returns the volmertic representation of example.
        c (int): label of a neurite in vol to use as source
        example_type (bool): True for positive example, False for negative example
        radius (int): radius (voxels) on bottom cross section in which to select second neurite
        num_slices (int): number of slices to drop
        context_slices (int): max number of slice on top and bottom neurites
    """

    def get_volumetric_example(self, vol, c, label, num_slices, radius, context_slices):

        margin = 1  # number of slices that must be left on top after droping slices
        top_c = c
        (sz, sy, sx) = vol.shape

        # Find min and max z slice on which c occurs #
        for i in range(sz):
            if c in vol[i]:
                zmin = i
                break
        for i in reversed(range(sz)):
            if c in vol[i]:
                zmax = i
                break
        assert zmax - zmin >= num_slices + \
            2, f'zspan of neurite must be at least 2 slices bigger than num_slices to drop, zspan:{zmax - zmin}, num_slices:{num_slices}'

        # the drop can start at the end of the top nerutie for negative examples
        # but should start earlier such that there is some bottom fragment for postive examples
        z_max_range = zmax-margin-num_slices+1 if label == 1 else zmax+1
        # margin not needed on bottom
        drop_start = random.randint(zmin+margin, z_max_range)
        # take min to ensure there is some bottom vol
        drop_end = min(drop_start+num_slices, vol.shape[0]-1)
        top_z_len = min(context_slices, drop_start-zmin)
        bot_z_len = min(context_slices, sz-drop_end)

        if self.verbose:
            print(
                f'num_slices: {num_slices}, drop: [{drop_start}, {drop_end}]')

        # Alloc final vol, we dont know how large it will be in y and x but we know max z #
        mz = num_slices + top_z_len + bot_z_len
        final_vol = np.zeros((mz, sy, sx), dtype='uint')

        # Build top section #
        top_vol_section = final_vol[0:top_z_len]
        top_vol_section[vol[drop_start-top_z_len:drop_start] == top_c] = top_c

        # Do connected component relabeling to ensure only one fragment on top #
        top_vol_section_relabeled = cc3d.connected_components(top_vol_section)
        # must select from top border slice
        top_classes_relabeled = list(np.unique(top_vol_section_relabeled[-1]))
        top_classes_relabeled = list_remove(top_classes_relabeled, 0)
        # select new top class from relabeled top section
        relabeled_top_c = random.choice(top_classes_relabeled)
        top_vol_section_relabeled[top_vol_section_relabeled !=
                                  relabeled_top_c] = 0

        # Get midpoint of neurite on 2D top cross section, #
        top_border = top_vol_section_relabeled[-1]
        # use the relabeled top section
        (com_x, com_y) = ndimage.measurements.center_of_mass(top_border)
        (com_x, com_y) = round(com_x), round(com_y)

        # Find all neurites with distnce D from that point on bottom cross section #
        bot_border = vol[drop_end].copy()  # need copy because we zero
        mask = circular_mask(
            bot_border.shape[0], bot_border.shape[1], center=(com_y, com_x), radius=radius)
        bot_border[~mask] = 0
        mismatch_classes = list(np.unique(bot_border))

        # For positive examples, simply set bottom class to top class #
        if label == 1:
            bot_c = top_c
        else:
            # Other wise select bottom class by picking 1 neurite from set of labels in radius #
            assert mismatch_classes[0] == 0, 'first class should be 0, otherwise something went wrong'
            # remove 0 and top class lables
            mismatch_classes = list_remove(mismatch_classes, [0, top_c])
            if len(mismatch_classes) == 0:
                if self.verbose:
                    print(
                        f'(find negative) for {label} example, class {c}, cut {drop_start, drop_end} could not find bottom label within radius, returning none')
                return None
            # maybe could select here based on on cross-sectional volume
            # select bottom neurite class
            bot_c = random.choice(mismatch_classes)

        # Build bot section #
        bot_vol_section = final_vol[num_slices+top_z_len:]
        bot_vol_section[vol[drop_end:drop_end+bot_z_len] == bot_c] = bot_c

        # Do connected component relabeling to ensure only one fragment on bottom #
        # The mask and radius are needed for both positive and negative examples #
        # So that after connected components we can pick a fragment near the top neurite #

        # plus minus one is hack to fix bug in cc3d https://github.com/seung-lab/connected-components-3d/issues/74
        bot_vol_section_relabeled = cc3d.connected_components(bot_vol_section)

        bot_border_relabled = bot_vol_section_relabeled[0]
        relabeled_fragments_in_radius = list(
            np.unique(bot_border_relabled[mask]))
        relabeled_fragments_in_radius = list_remove(
            relabeled_fragments_in_radius, 0)
        if len(relabeled_fragments_in_radius) == 0:
            if self.verbose:
                print(
                    f'(relabel bot) for {label} example, class {c}, cut {drop_start, drop_end}, could not find bottom label within radius, returning none')
            return None
        # take fragment which is in radius
        relabeled_bot_c = random.choice(relabeled_fragments_in_radius)
        bot_vol_section_relabeled[bot_vol_section_relabeled !=
                                  relabeled_bot_c] = 0

        # Build final volume of top and bottom sections #
        final_vol[0: top_z_len] = top_vol_section_relabeled
        final_vol[num_slices+top_z_len:] = bot_vol_section_relabeled

        return final_vol

    def get_vol_class_label_from_index(self, index):

        index = index % self.true_length
        label = np.int64(index % 2 == 0)  # label based on even or odd
        index = index // 2
        c = self.classes[index]
        vol_i = self.class_i_to_vol_i[index]
        vol = self.vols[vol_i]

        if self.verbose:
            print(f'{label}, vol: {vol_i}, c: {c}')

        return vol, c, label

    def convert_to_point_cloud(self, vol):

        pc = convert_grid_to_pointcloud(vol)

        if self.num_points is not None:
            num_points = pc.shape[0]

            if num_points < self.num_points:
                if self.verbose:
                    print(
                        f'not enough points, need {self.num_points}, have {num_points}, replace sampling to fix')

                pc = random_sample_arr(
                    pc, count=self.num_points, replace=True)

            else:
                pc = random_sample_arr(pc, count=self.num_points)

        return pc

    def remove_vol_interiors(self, vol):

        def rm_interior(v):
            return v * find_boundaries(
                v, mode='inner')

        if self.open_vol:
            for i in range(vol.shape[0]):
                vol[i] = rm_interior(vol[i])
        else:
            vol = rm_interior(vol)

        return vol

    def get_example(self, index):

        vol, c, label = self.get_vol_class_label_from_index(index)

        # choose num_slices
        num_slices = random.randint(self.num_slices[0], self.num_slices[1])
        vol_example = self.get_volumetric_example(
            vol, c, label, num_slices, self.radius, self.context_slices)

        # if we cant build example, try again with random index
        if self.retry and vol_example is None:
            rand_i = random.randint(0, self.effective_length)
            if self.verbose:
                print(f'redo on i {rand_i}')
            return self.get_example(rand_i)

        # final crop and relabel
        vol_example = crop_where(vol_example, vol_example != 0)
        vol_example = cc3d.connected_components(vol_example)

        # sanity check
        all_classes = np.unique(vol_example)
        assert len(
            all_classes) == 3, f'final sample should have 3 classes, [0, n1, n2] not {all_classes}'

        # remove interiors
        vol_example = self.remove_vol_interiors(vol_example)

        # convert to point cloud
        pc_example = self.convert_to_point_cloud(vol_example)
        if self.Augmentor is not None:
            pc_example = self.Augmentor.transfrom(pc_example)

        # swap axes if needed
        if self.swapaxes:
            pc_example = np.swapaxes(pc_example, 0, 1)

        if self.verbose:
            print(
                f'pc shape: {pc_example.shape}, vol shape: {vol_example.shape}')

        if self.torch:
            pc_example = torch.from_numpy(pc_example).type(torch.float32)
            label = torch.tensor(label).type(torch.LongTensor)

        if self.return_vol:
            return (pc_example, label, vol_example)

        return (pc_example, label)

    def __getitem__(self, index):

        # return torch.from_numpy(np.random.rand(3, self.num_points)).type(torch.float32), torch.tensor(np.int64(index % 2 == 0)).type(torch.LongTensor)
        return self.get_example(index)

    def __len__(self):
        return self.effective_length


if __name__ == '__main__':

    num_slices = [1, 4]
    radius = 96
    context_slices = 6
    num_points = 1024
    batch_size = 128
    num_workers = 128
    print('reading vols...')
    train_vols, test_vols = prepare_cremi_vols('./dataset/cremi')

    print('building dataset...')
    augmentor = Augmentor(center=True, shuffle=True, rotate=True, scale=True)
    dataset = SplitterDataset(train_vols, num_slices, radius, context_slices, num_points=num_points, torch=True, open_vol=True, Augmentor=augmentor)
    print(len(dataset))
    print('building dataloader...')
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=num_workers)
    print(len(dataloader))
        
    epochs = 5
    times = []
    for e in range(epochs):
        start = time.time()
        for i, batch in enumerate(dataloader):
            taken = time.time() - start
            
            times.append(taken)
            print(f'{i} took {taken}')
            start = time.time()
        
        
    # make_histogram(times, bins=len(dataloader)//4)
    print(min(times), max(times))