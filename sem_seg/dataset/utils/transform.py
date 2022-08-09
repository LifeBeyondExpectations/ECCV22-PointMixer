import numpy as np
import random
import torch
import scipy
import scipy.ndimage
import scipy.interpolate


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, coord, feat, label):
        for t in self.transforms:
            coord, feat, label = t(coord, feat, label)
        return coord, feat, label


class ToTensor(object):
    def __call__(self, coord, feat, label):
        coord = torch.from_numpy(coord)
        if not isinstance(coord, torch.FloatTensor):
            coord = coord.float()
        feat = torch.from_numpy(feat)
        if not isinstance(feat, torch.FloatTensor):
            feat = feat.float()
        label = torch.from_numpy(label)
        if not isinstance(label, torch.LongTensor):
            label = label.long()
        return coord, feat, label


class RandomRotate(object):
    def __init__(self, angle=[0, 0, 1]):
        self.angle = angle

    def __call__(self, coord, feat, label):
        angle_x = np.random.uniform(-self.angle[0], self.angle[0]) * np.pi
        angle_y = np.random.uniform(-self.angle[1], self.angle[1]) * np.pi
        angle_z = np.random.uniform(-self.angle[2], self.angle[2]) * np.pi
        cos_x, sin_x = np.cos(angle_x), np.sin(angle_x)
        cos_y, sin_y = np.cos(angle_y), np.sin(angle_y)
        cos_z, sin_z = np.cos(angle_z), np.sin(angle_z)
        R_x = np.array([[1, 0, 0], [0, cos_x, -sin_x], [0, sin_x, cos_x]])
        R_y = np.array([[cos_y, 0, sin_y], [0, 1, 0], [-sin_y, 0, cos_y]])
        R_z = np.array([[cos_z, -sin_z, 0], [sin_z, cos_z, 0], [0, 0, 1]])
        R = np.dot(R_z, np.dot(R_y, R_x))
        coord = np.dot(coord, np.transpose(R))
        return coord, feat, label


class CoordCrop(object):
    def __init__(self, npart=9):
        self.npart = int(npart)
        self.base = 3
        assert (self.npart % self.base) == 0, \
            ('npart[%d] should be in [%d,%d,%d,...]'%
            (self.npart, self.base, self.base**2, self.base**3))

        for _ in range(3):
            print(">> CoordCrop [%d]"%(self.npart))

    def __call__(self, coord, feat, label):
        shifted_coord = coord - coord.mean(axis=0)
        idx = random.randint(0, self.base**3)
        cond = np.ones([shifted_coord.shape[0]], dtype=np.bool)
        """ 
        0,1,2 / 0,1,2 / ...
        """
        if self.npart <= self.base:
            idx_cond = idx % self.base
            if idx_cond == 0:
                cond = np.logical_and(cond, shifted_coord[:,0]>0)
            elif idx_cond == 1:
                cond = np.logical_and(cond, shifted_coord[:,0]<0)
            # else:
            #     print("sample globally")
        
        """ 
        0,0,0 / 1,1,1 / 2,2,2 ...
        """
        if self.npart <= (self.base*2):
            idx_ = int(idx/self.base)
            idx_cond = idx_ % self.base
            if idx_cond == 0:
                cond = np.logical_and(cond, shifted_coord[:,1]>0)
            elif idx_cond == 1:
                cond = np.logical_and(cond, shifted_coord[:,1]<0)
            # else:
            #     print("sample globally")

        """ 
        0,0,0 / 0,0,0 / 0,0,0 / 1,1,1 / ...
        """
        if self.npart <= (self.base*3):
            idx_ = int(idx/int(self.base**2))
            idx_cond = idx_ % self.base
            if idx_cond == 0:
                cond = np.logical_and(cond, shifted_coord[:,2]>0)
            elif idx_cond == 1:
                cond = np.logical_and(cond, shifted_coord[:,2]<0)
            # else:
            #     print("sample globally")
        
        if cond.sum() < 5000:
            cond = np.ones([shifted_coord.shape[0]], dtype=np.bool)

        coord = coord[cond, :]
        feat = feat[cond, :]
        label = label[cond] # label.shape = [npts]
        return coord, feat, label


class RandomCrop(object):
    def __init__(self, x, y, z, application_ratio=1, max_retries=10):
        assert x > 0
        assert y > 0
        assert z > 0
        self.application_ratio = application_ratio
        self.max_size = np.array([[x, y, z]])
        self.max_retries = max_retries

    def __call__(self, coords: np.array, feats, labels):
        if random.random() > self.application_ratio:
            return coords, feats, labels

        norm_coords = coords - coords.min(0, keepdims=True)
        max_coords = norm_coords.max(0, keepdims=True)
        # start range
        coord_range = max_coords - self.max_size
        coord_range = np.clip(coord_range, a_min=0, a_max=float("inf"))
        # If crop size is larger than the coordinates, return orig
        if np.prod(coord_range == 0):
            return coords, feats, labels
        # sample crop start point
        valid = False
        retries = 0
        while not valid:
            min_box = np.random.rand(1, 3) * coord_range
            max_box = min_box + self.max_size
            sel = np.logical_and(np.prod(norm_coords > min_box, 1), np.prod(norm_coords < max_box, 1))
            if np.sum(sel) > 0: valid = True
            retries += 1
            if retries >= self.max_retries: break

        if valid:
            return \
                (
                    coords[sel],
                    feats if feats is None else feats[sel],
                    labels if labels is None else labels[sel],
                )
        else:
            return coords, feats, labels


class RandomScale(object):
    def __init__(self, scale=[0.9, 1.1], anisotropic=False):
        self.scale = scale
        self.anisotropic = anisotropic

    def __call__(self, coord, feat, label):
        scale = np.random.uniform(self.scale[0], self.scale[1], 3 if self.anisotropic else 1)
        coord *= scale
        return coord, feat, label


class RandomShift(object):
    def __init__(self, shift=[0.2, 0.2, 0]):
        self.shift = shift

    def __call__(self, coord, feat, label):
        shift_x = np.random.uniform(-self.shift[0], self.shift[0])
        shift_y = np.random.uniform(-self.shift[1], self.shift[1])
        shift_z = np.random.uniform(-self.shift[2], self.shift[2])
        coord += [shift_x, shift_y, shift_z]
        return coord, feat, label


class RandomFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, coord, feat, label):
        if np.random.rand() < self.p:
            coord[:, 0] = -coord[:, 0]
        if np.random.rand() < self.p:
            coord[:, 1] = -coord[:, 1]
        return coord, feat, label


class RandomJitter(object):
    def __init__(self, sigma=0.01, clip=0.05):
        self.sigma = sigma
        self.clip = clip

    def __call__(self, coord, feat, label):
        assert (self.clip > 0)
        jitter = np.clip(self.sigma * np.random.randn(coord.shape[0], 3), -1 * self.clip, self.clip)
        coord += jitter
        return coord, feat, label


class ChromaticAutoContrast(object):
    def __init__(self, p=0.2, blend_factor=None):
        self.p = p
        self.blend_factor = blend_factor

    def __call__(self, coord, feat, label):
        if np.random.rand() < self.p:
            lo = np.min(feat, 0, keepdims=True)
            hi = np.max(feat, 0, keepdims=True)
            scale = 255 / (hi - lo)
            contrast_feat = (feat[:, :3] - lo) * scale
            blend_factor = np.random.rand() if self.blend_factor is None else self.blend_factor
            feat[:, :3] = (1 - blend_factor) * feat[:, :3] + blend_factor * contrast_feat
        return coord, feat, label


class ChromaticTranslation(object):
    def __init__(self, p=0.95, ratio=0.05):
        self.p = p
        self.ratio = ratio

    def __call__(self, coord, feat, label):
        if np.random.rand() < self.p:
            tr = (np.random.rand(1, 3) - 0.5) * 255 * 2 * self.ratio
            feat[:, :3] = np.clip(tr + feat[:, :3], 0, 255)
        return coord, feat, label


class ChromaticJitter(object):
    def __init__(self, p=0.95, std=0.005):
        self.p = p
        self.std = std

    def __call__(self, coord, feat, label):
        if np.random.rand() < self.p:
            noise = np.random.randn(feat.shape[0], 3)
            noise *= self.std * 255
            feat[:, :3] = np.clip(noise + feat[:, :3], 0, 255)
        return coord, feat, label


class HueSaturationTranslation(object):
    @staticmethod
    def rgb_to_hsv(rgb):
        # Translated from source of colorsys.rgb_to_hsv
        # r,g,b should be a numpy arrays with values between 0 and 255
        # rgb_to_hsv returns an array of floats between 0.0 and 1.0.
        rgb = rgb.astype('float')
        hsv = np.zeros_like(rgb)
        # in case an RGBA array was passed, just copy the A channel
        hsv[..., 3:] = rgb[..., 3:]
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        maxc = np.max(rgb[..., :3], axis=-1)
        minc = np.min(rgb[..., :3], axis=-1)
        hsv[..., 2] = maxc
        mask = maxc != minc
        hsv[mask, 1] = (maxc - minc)[mask] / maxc[mask]
        rc = np.zeros_like(r)
        gc = np.zeros_like(g)
        bc = np.zeros_like(b)
        rc[mask] = (maxc - r)[mask] / (maxc - minc)[mask]
        gc[mask] = (maxc - g)[mask] / (maxc - minc)[mask]
        bc[mask] = (maxc - b)[mask] / (maxc - minc)[mask]
        hsv[..., 0] = np.select([r == maxc, g == maxc], [bc - gc, 2.0 + rc - bc], default=4.0 + gc - rc)
        hsv[..., 0] = (hsv[..., 0] / 6.0) % 1.0
        return hsv

    @staticmethod
    def hsv_to_rgb(hsv):
        # Translated from source of colorsys.hsv_to_rgb
        # h,s should be a numpy arrays with values between 0.0 and 1.0
        # v should be a numpy array with values between 0.0 and 255.0
        # hsv_to_rgb returns an array of uints between 0 and 255.
        rgb = np.empty_like(hsv)
        rgb[..., 3:] = hsv[..., 3:]
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        i = (h * 6.0).astype('uint8')
        f = (h * 6.0) - i
        p = v * (1.0 - s)
        q = v * (1.0 - s * f)
        t = v * (1.0 - s * (1.0 - f))
        i = i % 6
        conditions = [s == 0.0, i == 1, i == 2, i == 3, i == 4, i == 5]
        rgb[..., 0] = np.select(conditions, [v, q, p, p, t, v], default=v)
        rgb[..., 1] = np.select(conditions, [v, v, v, q, p, p], default=t)
        rgb[..., 2] = np.select(conditions, [v, p, t, v, v, q], default=p)
        return rgb.astype('uint8')

    def __init__(self, hue_max=0.5, saturation_max=0.2):
        self.hue_max = hue_max
        self.saturation_max = saturation_max

    def __call__(self, coord, feat, label):
        # Assume feat[:, :3] is rgb
        hsv = HueSaturationTranslation.rgb_to_hsv(feat[:, :3])
        hue_val = (np.random.rand() - 0.5) * 2 * self.hue_max
        sat_ratio = 1 + (np.random.rand() - 0.5) * 2 * self.saturation_max
        hsv[..., 0] = np.remainder(hue_val + hsv[..., 0] + 1, 1)
        hsv[..., 1] = np.clip(sat_ratio * hsv[..., 1], 0, 1)
        feat[:, :3] = np.clip(HueSaturationTranslation.hsv_to_rgb(hsv), 0, 255)
        return coord, feat, label


class RandomDropColor(object):
    def __init__(self, p=0.2):
        self.p = p

    def __call__(self, coord, feat, label):
        if np.random.rand() < self.p:
            feat[:, :3] = 0
            # feat[:, :3] = 127.5
        return coord, feat, label

##############################
# Augmentation from Minkowski engine (Choy et al., 2020)
##############################
class ElasticDistortion:
  def __init__(self, distortion_params=((0.2, 0.4), (0.8, 1.6))):
    self.distortion_params = distortion_params

  def elastic_distortion(self, coords, feats, labels, granularity, magnitude):
    """Apply elastic distortion on sparse coordinate space.
      pointcloud: numpy array of (number of points, at least 3 spatial dims)
      granularity: size of the noise grid (in same scale[m/cm] as the voxel grid)
      magnitude: noise multiplier
    """
    blurx = np.ones((3, 1, 1, 1)).astype('float32') / 3
    blury = np.ones((1, 3, 1, 1)).astype('float32') / 3
    blurz = np.ones((1, 1, 3, 1)).astype('float32') / 3
    coords_min = coords.min(0)

    # Create Gaussian noise tensor of the size given by granularity.
    noise_dim = ((coords - coords_min).max(0) // granularity).astype(int) + 3
    noise = np.random.randn(*noise_dim, 3).astype(np.float32)

    # Smoothing.
    for _ in range(2):
      noise = scipy.ndimage.filters.convolve(noise, blurx, mode='constant', cval=0)
      noise = scipy.ndimage.filters.convolve(noise, blury, mode='constant', cval=0)
      noise = scipy.ndimage.filters.convolve(noise, blurz, mode='constant', cval=0)

    # Trilinear interpolate noise filters for each spatial dimensions.
    ax = [
        np.linspace(d_min, d_max, d)
        for d_min, d_max, d in zip(coords_min - granularity, coords_min + granularity *
                                   (noise_dim - 2), noise_dim)
    ]
    interp = scipy.interpolate.RegularGridInterpolator(ax, noise, bounds_error=0, fill_value=0)
    coords += interp(coords) * magnitude
    return coords, feats, labels

  def __call__(self, coords, feats, labels):
    if self.distortion_params is not None:
      if random.random() < 0.95:
        for granularity, magnitude in self.distortion_params:
          coords, feats, labels = self.elastic_distortion(coords, feats, labels, granularity,
                                                          magnitude)
    return coords, feats, labels

class RandomDropout(object):
  def __init__(self, dropout_ratio=0.2, dropout_application_ratio=0.5):
    """
    upright_axis: axis index among x,y,z, i.e. 2 for z
    """
    self.dropout_ratio = dropout_ratio
    self.dropout_application_ratio = dropout_application_ratio

  def __call__(self, coords, feats, labels):
    if random.random() < self.dropout_ratio:
      N = len(coords)
      inds = np.random.choice(N, int(N * (1 - self.dropout_ratio)), replace=False)
      return coords[inds], feats[inds], labels[inds]
    return coords, feats, labels

class RandomHorizontalFlip(object):
  def __init__(self, upright_axis, is_temporal=False):
    """
    upright_axis: axis index among x,y,z, i.e. 2 for z
    """
    self.is_temporal = is_temporal
    self.D = 4 if is_temporal else 3
    self.upright_axis = {'x': 0, 'y': 1, 'z': 2}[upright_axis.lower()]
    # Use the rest of axes for flipping.
    self.horz_axes = set(range(self.D)) - set([self.upright_axis])

  def __call__(self, coords, feats, labels):
    if random.random() < 0.95:
      for curr_ax in self.horz_axes:
        if random.random() < 0.5:
          coord_max = np.max(coords[:, curr_ax])
          coords[:, curr_ax] = coord_max - coords[:, curr_ax]
    return coords, feats, labels