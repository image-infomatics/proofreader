import numpy as np
from .all import split_int


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
    crop_slice = []
    for cord in np.where(condition):
        amin, amax = np.min(cord), np.max(cord)
        crop_slice.append(slice(amin, amax))
    crop_slice = tuple(crop_slice)
    cropped = vol[crop_slice]
    return cropped

# get the min and max indices where conditions is true


def arg_where_range(condition):
    where_indices = np.argwhere(condition)
    mins, maxs = np.amin(where_indices, axis=0), np.amax(where_indices, axis=0)
    return mins, maxs

 # return a circular mask at center with radius
 # https://newbedev.com/how-can-i-create-a-circular-mask-for-a-numpy-array


def circular_mask(h, w, center=None, radius=None):

    if center is None:  # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None:  # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask
