import numpy as np


"""
SOME CODE BORROWED FROM: https://github.com/charlesq34/pointnet2/blob/master/utils/provider.py
"""


class Augmentor(object):
    def __init__(self,
                 shuffle: bool = False,
                 center: bool = False,
                 rotate: bool = False,
                 scale: bool = False,
                 jitter: bool = False,
                 normalize: tuple = [125, 1250, 1250]
                 ):
        """
        Used to augment single pointcloud example
        """
        self.shuffle = shuffle
        self.center = center
        self.rotate = rotate
        self.scale = scale
        self.jitter = jitter
        self.normalize = normalize

    def transfrom(self, data):
        """ Apply transforms to pointcloud data
        Input:
            NxC array
        Output:
            NxC array
        """
        assert data.shape[0] > data.shape[1], 'data should be size POINTSxDIMS'

        if self.normalize is not None:
            data = normalize_point_cloud(data, self.normalize)

        if self.rotate:
            data = rotate_point_cloud(data)

        if self.scale:
            data = random_scale_point_cloud(data)

        if self.jitter:
            data = jitter_point_cloud(data)

        if self.center:
            data = center_pointcloud(data)

        if self.shuffle:
            data = shuffle_points(data)

        return data


def normalize_point_cloud(data, factor):
    """ Normalize data by factor amount in each dim
        Input:
            NxC array
        Output:
            NxC array
    """
    data[:] /= np.array(factor)
    return data


def shuffle_points(data):
    """ Shuffle orders of points in each point cloud -- changes FPS behavior.
        Use the same shuffling idx for the entire batch.
        Input:
            NxC array
        Output:
            NxC array
    """
    idx = np.arange(data.shape[0])
    np.random.shuffle(idx)
    return data[idx, :]


def center_pointcloud(data, center=(0, 0, 0)):
    """ Centers the point cloud so the midpoint is at center
        Input:
            NxC array
        Output:
            NxC array
    """
    mps = (np.amin(data, axis=0, keepdims=True) +
           np.amax(data, axis=0, keepdims=True))/2
    data[:] -= mps - center
    return data


def rotate_point_cloud(data):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          Nx3 array, original batch of point clouds
        Return:
          Nx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(data.shape, dtype=np.float32)
    rotation_angle = np.random.uniform() * 2 * np.pi
    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)
    rotation_matrix = np.array([[cosval, 0, sinval],
                                [0, 1, 0],
                                [-sinval, 0, cosval]])
    rotated_data = np.dot(data.reshape((-1, 3)), rotation_matrix)

    return rotated_data


def rotate_point_cloud_z(data):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          Nx3 array, original batch of point clouds
        Return:
          Nx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(data.shape, dtype=np.float32)
    rotation_angle = np.random.uniform() * 2 * np.pi
    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)
    rotation_matrix = np.array([[cosval, sinval, 0],
                                [-sinval, cosval, 0],
                                [0, 0, 1]])
    rotated_data = np.dot(
        data.reshape((-1, 3)), rotation_matrix)
    return rotated_data


def rotate_perturbation_point_cloud(data, angle_sigma=0.06, angle_clip=0.18):
    """ Randomly perturb the point clouds by small rotations
        Input:
          Nx3 array, original batch of point clouds
        Return:
          Nx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(data.shape, dtype=np.float32)

    angles = np.clip(angle_sigma*np.random.randn(3), -
                     angle_clip, angle_clip)
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(angles[0]), -np.sin(angles[0])],
                   [0, np.sin(angles[0]), np.cos(angles[0])]])
    Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                   [0, 1, 0],
                   [-np.sin(angles[1]), 0, np.cos(angles[1])]])
    Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                   [np.sin(angles[2]), np.cos(angles[2]), 0],
                   [0, 0, 1]])
    R = np.dot(Rz, np.dot(Ry, Rx))
    shape_pc = data
    rotated_data = np.dot(shape_pc.reshape((-1, 3)), R)

    return rotated_data


def jitter_point_cloud(data, sigma=0.01, clip=0.05):
    """ Randomly jitter points. jittering is per point.
        Input:
          Nx3 array, original batch of point clouds
        Return:
          Nx3 array, jittered batch of point clouds
    """
    N, C = data.shape
    assert(clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(N, C), -1*clip, clip)
    jittered_data += data
    return jittered_data


def random_scale_point_cloud(data, scale_low=0.9, scale_high=1.1):
    """ Randomly scale the point cloud. Scale is per point cloud.
        Input:
            Nx3 array, original batch of point clouds
        Return:
            Nx3 array, scaled batch of point clouds
    """
    scales = np.random.uniform(scale_low, scale_high)
    data *= scales

    return data
