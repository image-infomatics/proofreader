from scipy.ndimage.morphology import distance_transform_edt
import numpy as np
import math


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


def get_unique_sorted_by_count(vol, reverse=False, return_counts=False):

    classes, counts = np.unique(vol, return_counts=True)

    sort_indices = np.argsort(counts)
    if reverse:
        sort_indices = np.flip(sort_indices)
    classes = classes[sort_indices]
    if return_counts:
        counts = counts[sort_indices]
        return classes, counts
    return classes


def crop_where(vol, condition):
    """
    Crop a matrix where the given conditions is true
    """
    crop_slice = []
    for cord in np.where(condition):
        amin, amax = np.min(cord), np.max(cord)
        crop_slice.append(slice(amin, amax+1))
    crop_slice = tuple(crop_slice)
    cropped = vol[crop_slice]
    return cropped


def arg_where_range(condition):
    """
    Get the min and max indices where conditions is true
    """
    where_indices = np.argwhere(condition)
    mins, maxs = np.amin(where_indices, axis=0), np.amax(where_indices, axis=0)
    return mins, maxs


def circular_mask(h, w, center=None, radius=None):
    """
    Return a circular mask at center with radius
    https://newbedev.com/how-can-i-create-a-circular-mask-for-a-numpy-array
    """
    if center is None:  # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None:  # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask


def get_grid_from_matrix(mat):
    """
    Get coordinate grid from matrix
    """
    sl = []
    for d in mat.shape:
        sl.append(slice(0, d))
    sl = tuple(sl)
    return np.mgrid[sl]


def cartesian_product(*arrays):
    """
    Return the Cartesian product of n arrays
    https://stackoverflow.com/questions/11144513/cartesian-product-of-x-and-y-array-points-into-single-array-of-2d-points
    """
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[..., i] = a
    return arr.reshape(-1, la)


def random_sample_arr(arr, ratio=None, count=None, replace=False):
    """
    Randomly sample a given array
    """
    rnone = ratio is None
    cnone = count is None
    assert rnone != cnone, 'must specify ratio xor count'
    amount = 0
    if not rnone:
        amount = round(len(arr)*ratio)
    else:
        amount = count
    r_ind = np.random.choice(len(arr), amount, replace=replace)
    return arr[r_ind]


def equivariant_shuffle(*arrays):
    array_lens = arrays[0].shape[0]
    shuffler = np.random.permutation(array_lens)
    res = []
    for a in arrays:
        res.append(a[shuffler])

    return tuple(res)


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


def zero_classes_with_zspan_less_than(vol, span, zero_val=0):
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
        if cnt < span:
            res.append(c)
    res = np.array(res)
    mask = np.isin(vol, res)
    new_vol = vol.copy()
    new_vol[mask] = zero_val

    return new_vol


def get_classes_with_at_least_volume(vol, min_volume):
    classes, counts = get_classes_sorted_by_volume(
        vol, return_counts=True, reverse=True)
    for i, cnt in enumerate(counts):
        if cnt < min_volume:
            break
    return classes[:i]


def zero_classes_with_min_volume(vol, min_volume, zero_val=0):
    classes, counts = get_classes_sorted_by_volume(
        vol, return_counts=True)
    i = 0
    for cnt in counts:
        if cnt > min_volume:
            break
        i += 1
    mask = np.isin(vol,  classes[:i])
    new_vol = vol.copy()
    new_vol[mask] = zero_val
    return new_vol


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


def correspond_labels(key, val, bg_label=0):
    res = {}
    classes = np.unique(key)
    for c in classes:
        if c != bg_label:
            corr = np.unique(val[key == c])
            # if len(corr) > 1:
            #     print('warn, multiple correspondance')
            res[c] = corr[-1]
    return res


def get_classes_sorted_by_distance(vol, distance_class, return_distances=False, reverse=False, ignore_zero=True, method='min'):

    methods = ['min', 'mean', 'max']
    assert method in methods, f'method should be one of {methods}'

    # set zero to some new label
    new_bg_label = np.max(vol) + 1
    vol[vol == 0] = new_bg_label
    # set object which we want to compute disantce from to zero
    vol[vol == distance_class] = 0
    distance_vol = distance_transform_edt(vol)
    classes = np.unique(vol)

    # redo vol so its not altered
    vol[vol == 0] = distance_class
    vol[vol == new_bg_label] = 0

    if ignore_zero:
        # zero is new_bg_label so this removes zero
        classes = list_remove(classes, new_bg_label)

    # distance_class is zero so this removes distance_class
    classes = list_remove(classes, 0)

    canidates = np.zeros(len(classes), dtype='uint')
    distances = np.zeros(len(classes))
    for i, c in enumerate(classes):
        if method == 'min':
            d = np.min(distance_vol[vol == c])
        elif method == 'mean':
            d = np.mean(distance_vol[vol == c])
        elif method == 'max':
            d = np.max(distance_vol[vol == c])
        canidates[i] = c
        distances[i] = d

    sort_indices = np.argsort(distances)
    if reverse:
        sort_indices = np.flip(sort_indices)

    # sort by distance
    if return_distances:
        return canidates[sort_indices], distances[sort_indices]
    else:
        return canidates[sort_indices]


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
