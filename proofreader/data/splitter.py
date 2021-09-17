from typing import List
import numpy as np
import torch
import random
from einops import rearrange
import cc3d
from proofreader.utils.all import list_remove, split_int
from proofreader.utils.data import random_sample_arr, circular_mask, crop_where
from proofreader.utils.vis import view_segmentation, grid_volume
from skimage.color import label2rgb
from skimage.segmentation import boundaries, find_boundaries
from scipy import ndimage


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
    # array of points representation
    cords = np.swapaxes(cords, 0, 1)
    return cords


class SplitterDataset(torch.utils.data.Dataset):
    def __init__(self,
                 vols: List,
                 num_slices: int,
                 radius: int,
                 context_slices: int,
                 fix_num_points: int = None,
                 open_vol: bool = True,
                 retry: bool = True,
                 verbose: bool = False,
                 return_vol: str = False
                 ):
        """
        Parameters:
            vols (List): list of vols to use.
            num_slices (int or 2-List[a,b]): the number of slices to drop, or range of slices to drop.
            radius (int): radius (voxels) on bottom cross section in which to select second neurite.
            context_slices (int): max number of slice on top and bottom neurites.
            fix_num_points (int): ensures number of points in pointcloud is exactly this.
            open_vol (bool): whether to get the exterior of the vol as open tube-like manifolds (True) or as closed balloon-like manifolds (False)
                             amounts to removing the interior along the z-axis (True) or for entire vol at once (False).
            retry (bool): whether to retry until success if cannot generate example
            verbose (bool): print warning messages.
            return_vol (str): whether to also return the volumetirc representation.
        """

        super().__init__()

        if isinstance(num_slices, int):
            num_slices = [num_slices, num_slices]
        else:
            assert len(
                num_slices) == 2, 'if num_slices is list must be len == 2 in indicating range.'

        self.vols = vols
        self.classes = []
        self.num_slices = num_slices
        self.radius = radius
        self.context_slices = context_slices
        self.fix_num_points = fix_num_points
        self.open_vol = open_vol
        self.retry = retry
        self.verbose = verbose
        self.return_vol = return_vol

        # get classes to use
        length = 0
        for vol in vols:
            # TODO figure out zspan condition other than max num_slices
            classes_z = get_classes_which_zspan_at_least(vol, num_slices[1]+2)
            classes_vol = get_classes_with_at_least_volume(vol, 400)
            cls_union = list(set(classes_z) & set(classes_vol))
            length += len(cls_union)
            self.classes.append(cls_union)

        # double for positive and negative examples
        self.length = length*2

    """
    Gets a postive or negative example from vol using some class seed. Returns the volmertic representation of example.
        c (int): label of a neurite in vol to use as source
        example_type (bool): True for positive example, False for negative example
        radius (int): radius (voxels) on bottom cross section in which to select second neurite
        num_slices (int): number of slices to drop
        context_slices (int): max number of slice on top and bottom neurites
    """

    def get_volumetric_example(self, vol, c, example_type, num_slices, radius, context_slices):

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
        z_max_range = zmax-margin-num_slices+1 if example_type else zmax+1
        # margin not needed on bottom
        drop_start = random.randint(zmin+margin, z_max_range)

        # take min to ensure there is some bottom vol
        drop_end = min(drop_start+num_slices, vol.shape[0]-1)
        top_z_len = min(context_slices, drop_start-zmin)
        bot_z_len = min(context_slices, sz-drop_end)

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
        if example_type:
            bot_c = top_c
        else:
            # Other wise select bottom class by picking 1 neurite from set of labels in radius #
            assert mismatch_classes[0] == 0, 'first class should be 0, otherwise something went wrong'
            # remove 0 and top class lables
            mismatch_classes = list_remove(mismatch_classes, [0, top_c])
            if len(mismatch_classes) == 0:
                if self.verbose:
                    print(
                        f'(find negative) for {example_type} example, class {c}, cut {drop_start, drop_end} could not find bottom label within radius, returning none')
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
                    f'(relabel bot) for {example_type} example, class {c}, cut {drop_start, drop_end}, could not find bottom label within radius, returning none')
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
        # determine which vol index is in
        for voli, classes_vol in enumerate(self.classes):
            # we double length to account for +/- examples for each c
            len_classes = len(classes_vol) * 2
            if index - len_classes < 0:
                break
            index -= len_classes

        label = index % 2 == 0  # label based on even or odd
        vol = self.vols[voli]
        index = index // 2
        c = self.classes[voli][index]
        return vol, c, label

    def convert_to_point_cloud(self, vol):

        pc = convert_grid_to_pointcloud(vol)

        if self.fix_num_points is not None:
            num_points = pc.shape[0]

            # need way to handle, could just retry?
            if num_points < self.fix_num_points:
                if self.verbose:
                    print(
                        f'not enough points, need {self.fix_num_points}, have {num_points}')
                return pc

            pc = random_sample_arr(pc, count=self.fix_num_points)

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
            rand_i = random.randint(0, self.length)
            if self.verbose:
                print(f'redo on i {rand_i}')
            return self.get_example(rand_i)

        # final crop and relabel
        vol_example = crop_where(vol_example, vol_example != 0)
        vol_example = cc3d.connected_components(vol_example)

        # sanity check
        labels = np.unique(vol_example)
        assert len(
            labels) == 3, f'final sample should have 3 labels, [0, n1, n2] not {labels}'

        # remove interiors
        vol_example = self.remove_vol_interiors(vol_example)

        # convert to point cloud
        pc_example = self.convert_to_point_cloud(vol_example)

        if self.return_vol:
            return (pc_example, vol_example, label)

        return (pc_example, label)

    def __getitem__(self, index):
        return self.get_example(index)

    def __len__(self):
        return self.length
