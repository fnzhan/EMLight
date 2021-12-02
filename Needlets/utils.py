import numpy as np
import OpenEXR
import cv2
import Imath
import math
from scipy import interpolate
# import torch
import healpy as hp

def convert_to_panorama(dirs, sizes, colors):
    grid_latitude, grid_longitude = torch.meshgrid(
        [torch.arange(128, dtype=torch.float), torch.arange(2 * 128, dtype=torch.float)])
    grid_latitude = grid_latitude.add(0.5)
    grid_longitude = grid_longitude.add(0.5)
    grid_latitude = grid_latitude.mul(np.pi / 128)
    grid_longitude = grid_longitude.mul(np.pi / 128)

    x = torch.sin(grid_latitude) * torch.cos(grid_longitude)
    y = torch.sin(grid_latitude) * torch.sin(grid_longitude)
    z = torch.cos(grid_latitude)
    xyz =  torch.stack((x, y, z))

    nbatch = colors.shape[0]
    lights = torch.zeros((nbatch, 3, 128, 256), dtype=dirs.dtype, device=dirs.device)
    _, tmp = colors.shape
    nlights = int(tmp / 3)
    for i in range(nlights):
        lights = lights + (colors[:, 3 * i + 0:3 * i + 3][:, :, None, None]) * (
            torch.exp(
                (torch.matmul(dirs[:, 3 * i + 0:3 * i + 3], xyz.view(3, -1)).
                 view(-1, xyz.shape[1], xyz.shape[2]) - 1) /
                (sizes[:, i]).view(-1, 1, 1))[:, None, :, :])
    return lights

def getSolidAngle(y, width, is3D=False):
    """
	y = y pixel position (cast as a float)
	Solid angles in latitude-longitude maps:
	http://webstaff.itn.liu.se/~jonun/web/teaching/2012-TNM089/Labs/IBL/scalefactors.pdf
	"""
    height = int(width / 2)
    pi2OverWidth = (np.pi * 2) / width
    piOverHeight = np.pi / height
    theta = (1.0 - ((y + 0.5) / height)) * np.pi
    return pi2OverWidth * (np.cos(theta - (piOverHeight / 2.0)) - np.cos(theta + (piOverHeight / 2.0)))


def getSolidAngleMap(width):
    height = int(width / 2)
    return np.repeat(getSolidAngle(np.arange(0, height), width)[:, np.newaxis], width, axis=1)


def tonemapping(im):
    power_im = np.power(im, 1 / 2.4)
    # print (np.amax(power_im))
    non_zero = power_im > 0
    if non_zero.any():
        r_percentile = np.percentile(power_im[non_zero], 95)
    else:
        r_percentile = np.percentile(power_im, 95)
    alpha = 0.95 / (r_percentile + 1e-10)
    tonemapped_im = np.multiply(alpha, power_im)

    tonemapped_im = np.clip(tonemapped_im, 0, 1)
    return tonemapped_im

def xy2ll(x, y, width, height):
    def yLocToLat(yLoc, height):
        return (yLoc / (float(height)/np.pi))
    def xLocToLon(xLoc, width):
        return (xLoc / (float(width)/(np.pi * 2)))
    return np.asarray([yLocToLat(y, height), xLocToLon(x, width)])

class PanoramaHandler(object):
    def __init__(self):
        super(PanoramaHandler, self).__init__()

    @staticmethod
    def rgb_to_intenisty(rgbs):
        intensity = 0.2126 * rgbs[..., 0] + 0.7152 * rgbs[..., 1] + 0.0722 * rgbs[..., 0]
        return intensity

    @staticmethod
    def read_exr(exr_path):
        File = OpenEXR.InputFile(exr_path)
        PixType = Imath.PixelType(Imath.PixelType.FLOAT)
        DW = File.header()['dataWindow']
        Size = (DW.max.x - DW.min.x + 1, DW.max.y - DW.min.y + 1)
        rgb = [np.fromstring(File.channel(c, PixType), dtype=np.float32) for c in 'RGB']
        r = np.reshape(rgb[0], (Size[1], Size[0]))
        g = np.reshape(rgb[1], (Size[1], Size[0]))
        b = np.reshape(rgb[2], (Size[1], Size[0]))
        hdr = np.zeros((Size[1], Size[0], 3), dtype=np.float32)
        hdr[:, :, 0] = r
        hdr[:, :, 1] = g
        hdr[:, :, 2] = b
        return hdr

    @staticmethod
    def read_hdr(hdr_path):
        hdr_img = cv2.imread(hdr_path, flags=cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        hdr_img = hdr_img[..., ::-1]
        return hdr_img

    @staticmethod
    def horizontal_rotate_panorama(hdr_img, deg):
        shift = int(deg / 360.0 * hdr_img.shape[1])
        out_img = np.roll(hdr_img, shift=shift, axis=1)
        return out_img

    @staticmethod
    def generate_steradian(height, width, multiply=True):
        steradian = np.linspace(0, height, num=height, endpoint=False) + 0.5
        steradian = np.sin(steradian / height * np.pi)
        steradian = np.tile(steradian.transpose(), (width, 1))
        steradian = steradian.transpose()
        if multiply:
            pixel_area = ((2 * np.pi) / width) * ((1 * np.pi) / height)
            steradian *= pixel_area
        return steradian.astype(np.float32)

    @staticmethod
    def prepare_gt_panorama(hdr_img, threshold=None):
        weight = PanoramaHandler.generate_steradian(height=hdr_img.shape[0], width=hdr_img.shape[1])
        hdr_intensity = PanoramaHandler.rgb_to_intenisty(hdr_img)

        if threshold is None or threshold < 0.0:
            threshold = hdr_intensity.max() / 20.

        mask = np.where(hdr_intensity < threshold)

        if mask[0].size != 0:
            ambient = np.sum(hdr_img[mask] * weight[mask][:, np.newaxis], axis=0, dtype=np.float32) / np.sum(
                weight[mask],
                dtype=np.float32)
        else:
            ambient = np.zeros([3], dtype=np.float32)

        hdr_img[mask] = 0.0
        return hdr_img, ambient

    @staticmethod
    def resize_panorama(hdr_img, new_shape):
        if isinstance(new_shape, tuple) and len(new_shape) == 2:
            hdr_img = cv2.resize(hdr_img, new_shape, interpolation=cv2.INTER_AREA)
        elif isinstance(new_shape, int):
            hdr_img = cv2.resize(hdr_img, (2 * new_shape, new_shape), interpolation=cv2.INTER_AREA)
        return hdr_img

    @staticmethod
    def crop_panorama(hdr_img, fov_deg, crop_image_h=720, crop_image_aspect_ratio="4:3"):
        # img must be float
        if hdr_img.dtype == np.uint8:
            hdr_img = hdr_img / 255.0

        numerator, denominator = [int(x) for x in crop_image_aspect_ratio.split(":")]
        ratio = numerator / denominator
        crop_image_w = int(crop_image_h * ratio)

        # print(crop_image_w, crop_image_h)
        scl = np.tan(np.deg2rad(fov_deg) / 2)
        sample_x, sample_y = np.meshgrid(
            np.linspace(-scl, scl, crop_image_w),
            np.linspace(-scl / ratio, scl / ratio, crop_image_h)
        )

        r = np.sqrt(sample_y * sample_y + sample_x * sample_x + 1)
        sample_x /= r
        sample_y /= r
        sample_z = np.sqrt(1 - sample_y * sample_y - sample_x * sample_x)
        # convert to polar
        azimuth = np.arctan2(sample_x, sample_z)  # [-Pi, Pi]
        elevation = np.arcsin(sample_y)  # [-Pi/2, Pi/2]
        # print(azimuth, elevation)

        # normalize to [0, 1]
        x = (1 + azimuth / np.pi) / 2
        y = (1 + elevation / (np.pi / 2)) / 2

        img_h = hdr_img.shape[0]
        img_w = hdr_img.shape[1]
        x = x * img_w
        y = y * img_h

        my_interpolating_function = interpolate.RegularGridInterpolator(
            (np.arange(0, hdr_img.shape[0]), np.arange(0, hdr_img.shape[1])), hdr_img)
        points = np.c_[y.ravel(), x.ravel()]
        out = my_interpolating_function(points).reshape((x.shape[0], x.shape[1], -1))
        return out

def sphere_points(n=128):
    golden_angle = np.pi * (3 - np.sqrt(5))
    theta = golden_angle * np.arange(n)
    z = np.linspace(1 - 1.0 / n, 1.0 / n - 1, n)
    radius = np.sqrt(1 - z * z)

    points = np.zeros((n, 3))
    points[:, 0] = radius * np.cos(theta)
    points[:, 1] = radius * np.sin(theta)
    points[:, 2] = z
    # xyz = points
    # x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    return points

def spherical_sampling(jmax=None, sampling=None):
    jmax = 2
    B = 2.0
    Nside = 2 ** math.ceil(math.log(math.floor(B ** (jmax + 1)) / 2, 2))
    Npix = 12 * Nside ** 2
    dir = np.vstack((hp.pix2vec(Nside, range(Npix))[i] for i in range(3)))
    dir = dir.transpose((1, 0))
    N = dir.shape[0]

    sample_points = sphere_points(64)
    ln = sample_points.shape[0]

    ids_mat = np.zeros((ln, 3))
    for n in range(ln):
        coord = sample_points[n]
        dis = []
        for i in range(N):
            coord2 = dir[i]
            tmp = np.linalg.norm(coord2 - coord)
            dis.append(tmp)
        idx = np.argsort(np.array(dis))
        ids_mat[n] = idx[:3]
    return ((ids_mat).astype('int'))
        # print(ids_mat[n])

class TonemapHDR(object):
    """
        Tonemap HDR image globally. First, we find alpha that maps the (max(numpy_img) * percentile) to max_mapping.
        Then, we calculate I_out = alpha * I_in ^ (1/gamma)
        input : nd.array batch of images : [H, W, C]
        output : nd.array batch of images : [H, W, C]
    """

    def __init__(self, gamma=2.4, percentile=50, max_mapping=0.5):
        self.gamma = gamma
        self.percentile = percentile
        self.max_mapping = max_mapping  # the value to which alpha will map the (max(numpy_img) * percentile) to

    def __call__(self, numpy_img, clip=True, alpha=None, gamma=True):
        if gamma:
            power_numpy_img = np.power(numpy_img, 1 / self.gamma)
        else:
            power_numpy_img = numpy_img
        non_zero = power_numpy_img > 0
        if non_zero.any():
            r_percentile = np.percentile(power_numpy_img[non_zero], self.percentile)
        else:
            r_percentile = np.percentile(power_numpy_img, self.percentile)

        if alpha is None:
            alpha = self.max_mapping / (r_percentile + 1e-10)
        tonemapped_img = np.multiply(alpha, power_numpy_img)

        if clip:
            tonemapped_img = np.clip(tonemapped_img, 0, 1)

        return tonemapped_img.astype('float32'), alpha