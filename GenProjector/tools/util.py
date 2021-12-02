"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import re
import importlib
import torch
from argparse import Namespace
import numpy as np
from PIL import Image
import argparse
import dill as pickle
import cv2
import numpy as np
import argparse
import math
import OpenEXR, Imath
import os
import imageio
import vtk
from vtk.util import numpy_support
imageio.plugins.freeimage.download()


def save_obj(obj, name):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)

# returns a configuration for creating a generator
# |default_opt| should be the opt of the current experiment
# |**kwargs|: if any configuration should be overriden, it can be specified here


def copyconf(default_opt, **kwargs):
    conf = argparse.Namespace(**vars(default_opt))
    for key in kwargs:
        print(key, kwargs[key])
        setattr(conf, key, kwargs[key])
    return conf


def tile_images(imgs, picturesPerRow=4):
    """ Code borrowed from
    https://stackoverflow.com/questions/26521365/cleanly-tile-numpy-array-of-images-stored-in-a-flattened-1d-format/26521997
    """

    # Padding
    if imgs.shape[0] % picturesPerRow == 0:
        rowPadding = 0
    else:
        rowPadding = picturesPerRow - imgs.shape[0] % picturesPerRow
    if rowPadding > 0:
        imgs = np.concatenate([imgs, np.zeros((rowPadding, imgs.shape[1:]), dtype=imgs.dtype)], axis=0)

    # Tiling Loop (The conditionals are not necessary anymore)
    tiled = []
    for i in range(0, imgs.shape[0], picturesPerRow):
        tiled.append(np.concatenate([imgs[j] for j in range(i, i + picturesPerRow)], axis=1))

    tiled = np.concatenate(tiled, axis=0)
    return tiled


# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array
def tensor2im(image_tensor, imtype=np.uint8, normalize=True, tile=False):
    if isinstance(image_tensor, list):
        image_numpy = []
        for i in range(len(image_tensor)):
            image_numpy.append(tensor2im(image_tensor[i], imtype, normalize))
        return image_numpy

    if image_tensor.dim() == 4:
        # transform each image in the batch
        images_np = []
        for b in range(image_tensor.size(0)):
            one_image = image_tensor[b]
            one_image_np = tensor2im(one_image)
            images_np.append(one_image_np.reshape(1, *one_image_np.shape))
        images_np = np.concatenate(images_np, axis=0)
        if tile:
            images_tiled = tile_images(images_np)
            return images_tiled
        else:
            return images_np

    if image_tensor.dim() == 2:
        image_tensor = image_tensor.unsqueeze(0)
    image_numpy = image_tensor.detach().cpu().float().numpy()
    # if normalize:
    #     image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    # else:
    image_numpy = np.transpose(image_numpy, (1, 2, 0))
    # image_numpy = np.clip(image_numpy, 0, 255)
    if image_numpy.shape[2] == 1:
        image_numpy = image_numpy[:, :, 0]
    return image_numpy


def save_image(image_numpy, image_path, create_dir=False):
    if create_dir:
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
    if len(image_numpy.shape) == 2:
        image_numpy = np.expand_dims(image_numpy, axis=2)
    if image_numpy.shape[2] == 1:
        image_numpy = np.repeat(image_numpy, 3, 2)
    image_pil = Image.fromarray(image_numpy)

    # save to png
    image_pil.save(image_path.replace('.jpg', '.png'))


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [atoi(c) for c in re.split('(\d+)', text)]


def natural_sort(items):
    items.sort(key=natural_keys)


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def find_class_in_module(target_cls_name, module):
    target_cls_name = target_cls_name.replace('_', '').lower()
    clslib = importlib.import_module(module)
    cls = None
    for name, clsobj in clslib.__dict__.items():
        if name.lower() == target_cls_name:
            cls = clsobj

    if cls is None:
        print("In %s, there should be a class whose name matches %s in lowercase without underscore(_)" % (module, target_cls_name))
        exit(0)

    return cls


def save_network(net, label, epoch, opt):
    save_filename = '%s_net_%s.pth' % (epoch, label)
    save_path = os.path.join(opt.checkpoints_dir, opt.name, save_filename)
    torch.save(net.cpu().state_dict(), save_path)
    if len(opt.gpu_ids) and torch.cuda.is_available():
        net.cuda()


def load_network(net, label, epoch, opt):
    save_filename = '%s_net_%s.pth' % (epoch, label)

    print ('***********')
    print (save_filename)

    save_dir = os.path.join(opt.checkpoints_dir, opt.name)
    save_path = os.path.join(save_dir, save_filename)
    weights = torch.load(save_path)
    net.load_state_dict(weights)
    return net

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


def tonemapping(im, sv_path, gamma=2.4, percentile=50, max_mapping=0.5):
    # im = np.expm1(im)
    # hdr_path = sv_path.replace('.png', '.hdr')
    # hdr = im.astype('float32')
    # imageio.imwrite(hdr_path, hdr, format='hdr')

    power_im = np.power(im, 1 / gamma)
    # print (np.amax(power_im))
    non_zero = power_im > 0
    if non_zero.any():
        r_percentile = np.percentile(power_im[non_zero], percentile)
    else:
        r_percentile = np.percentile(power_im, percentile)
    alpha = max_mapping / (r_percentile + 1e-10)
    tonemapped_im = np.multiply(alpha, power_im)

    tonemapped_im = np.clip(tonemapped_im, 0, 1)
    hdr = tonemapped_im * 255.0
    hdr = Image.fromarray(hdr.astype('uint8'))
    hdr.save(sv_path)


def load_exr(in_file):
    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    golden = OpenEXR.InputFile(in_file)
    dw = golden.header()['dataWindow']
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

    redstr = golden.channel('R', pt)
    red = np.fromstring(redstr, dtype = np.float32)
    red.shape = (size[1], size[0]) # Numpy arrays are (row, col)

    greenstr = golden.channel('G', pt)
    green = np.fromstring(greenstr, dtype = np.float32)
    green.shape = (size[1], size[0]) # Numpy arrays are (row, col)

    bluestr = golden.channel('B', pt)
    blue = np.fromstring(bluestr, dtype = np.float32)
    blue.shape = (size[1], size[0]) # Numpy arrays are (row, col)

    img = np.zeros((size[1], size[0], 3), dtype=np.float32)
    img[:, :, 0] = red
    img[:, :, 1] = green
    img[:, :, 2] = blue
    return img

def write_exr(out_file, data):
    exr = OpenEXR.OutputFile(out_file, OpenEXR.Header(data.shape[1], data.shape[0]))
    red = data[:, :, 0]
    green = data[:, :, 1]
    blue = data[:, :, 2]
    exr.writePixels({'R': red.tostring(), 'G': green.tostring(), 'B': blue.tostring()})

def resize_exr(img, res_x=512, res_y=512):

    theta, phi, move = 0.0, 0.0, 0.0
    img_x = img.shape[0]
    img_y = img.shape[1]

    theta = theta / 180 * math.pi
    phi = phi / 180 * math.pi

    axis_y = math.cos(theta)
    axis_z = math.sin(theta)
    axis_x = 0

    # theta rotation matrix
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)
    theta_rot_mat = np.array([[1, 0, 0], \
            [0, cos_theta, -sin_theta], \
            [0, sin_theta, cos_theta]], dtype=np.float32)

    # phi rotation matrix
    cos_phi = math.cos(phi)
    sin_phi = -math.sin(phi)
    phi_rot_mat = np.array([[cos_phi + axis_x**2 * (1 - cos_phi), \
            axis_x * axis_y * (1 - cos_phi) - axis_z * sin_phi, \
            axis_x * axis_z * (1 - cos_phi) + axis_y * sin_phi], \
            [axis_y * axis_x * (1 - cos_phi) + axis_z * sin_phi, \
            cos_phi + axis_y**2 * (1 - cos_phi), \
            axis_y * axis_z * (1 - cos_phi) - axis_x * sin_phi], \
            [axis_z * axis_x * (1 - cos_phi) - axis_y * sin_phi, \
            axis_z * axis_y * (1 - cos_phi) + axis_x * sin_phi, \
            cos_phi + axis_z**2 * (1 - cos_phi)]], dtype=np.float32)

    indx = np.tile(np.array(np.arange(res_x), dtype=np.float32), (res_y, 1)).T
    indy = np.tile(np.array(np.arange(res_y), dtype=np.float32), (res_x, 1))

    map_x = np.sin(indx * math.pi / res_x - math.pi / 2)
    map_y = np.sin(indy * (2 * math.pi)/ res_y) * np.cos(indx * math.pi / res_x - math.pi / 2)
    map_z = -np.cos(indy * (2 * math.pi)/ res_y) * np.cos(indx * math.pi / res_x - math.pi / 2)

    ind = np.reshape(np.concatenate((np.expand_dims(map_x, 2), np.expand_dims(map_y, 2), \
            np.expand_dims(map_z, 2)), axis=2), [-1, 3]).T

    move_dir = np.array([0, 0, -1], dtype=np.float32)
    move_dir = theta_rot_mat.dot(move_dir)
    move_dir = phi_rot_mat.dot(move_dir)

    ind = theta_rot_mat.dot(ind)
    ind = phi_rot_mat.dot(ind)

    ind += np.tile(move * move_dir, (ind.shape[1], 1)).T

    vec_len = np.sqrt(np.sum(ind**2, axis=0))
    ind /= np.tile(vec_len, (3, 1))

    cur_phi = np.arcsin(ind[0, :])
    cur_theta = np.arctan2(ind[1, :], -ind[2, :])

    map_x = (cur_phi + math.pi/2) / math.pi * img_x
    map_y = cur_theta % (2 * math.pi) / (2 * math.pi) * img_y

    map_x = np.reshape(map_x, [res_x, res_y])
    map_y = np.reshape(map_y, [res_x, res_y])

    return cv2.remap(img, map_y, map_x, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)


def convert_to_panorama(dirs, sizes, dist):
    grid_latitude, grid_longitude = torch.meshgrid(
        [torch.arange(128, dtype=torch.float), torch.arange(2 * 128, dtype=torch.float)])
    grid_latitude = grid_latitude.add(0.5)
    grid_longitude = grid_longitude.add(0.5)
    grid_latitude = grid_latitude.mul(np.pi / 128)
    grid_longitude = grid_longitude.mul(np.pi / 128)

    x = torch.sin(grid_latitude) * torch.cos(grid_longitude)
    y = torch.sin(grid_latitude) * torch.sin(grid_longitude)
    z = torch.cos(grid_latitude)
    xyz =  torch.stack((x, y, z)).cuda()

    nbatch = dist.shape[0]
    lights = torch.zeros((nbatch, 3, 128, 256), dtype=dirs.dtype, device=dirs.device)
    _, tmp = dist.shape
    nlights = int(tmp / 3)
    for i in range(nlights):
        lights = lights + (dist[:, 3 * i: 3 * i + 3][:, :, None, None]) * (
            torch.exp(
                (torch.matmul(dirs[:, 3 * i + 0:3 * i + 3], xyz.view(3, -1)).
                 view(-1, xyz.shape[1], xyz.shape[2]) - 1) /
                (sizes[:, i]).view(-1, 1, 1))[:, None, :, :])
    return lights


def normalize_2_unit_sphere(pts):
    num_pts = pts.GetNumberOfPoints()
    # print("we have #{} pts".format(num_pts))
    for i in range(num_pts):
        tmp = list(pts.GetPoint(i))
        n = vtk.vtkMath.Normalize(tmp)
        pts.SetPoint(i, tmp)

def polyhedron(subdivide=1):

    icosa = vtk.vtkPlatonicSolidSource()
    icosa.SetSolidTypeToIcosahedron()
    icosa.Update()
    subdivided_sphere = icosa.GetOutput()

    for i in range(subdivide):
        linear = vtk.vtkLinearSubdivisionFilter()
        linear.SetInputData(subdivided_sphere)
        linear.SetNumberOfSubdivisions(1)
        linear.Update()
        subdivided_sphere = linear.GetOutput()
        normalize_2_unit_sphere(subdivided_sphere.GetPoints())
        subdivided_sphere.Modified()

    # if save_directions:
    transform = vtk.vtkSphericalTransform()
    transform = transform.GetInverse()
    pts = subdivided_sphere.GetPoints()
    pts_spherical = vtk.vtkPoints()
    transform.TransformPoints(pts, pts_spherical)

    pts_arr = numpy_support.vtk_to_numpy(pts.GetData())
    # print (as_numpy.shape)
    return pts_arr


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
        rgba = [np.fromstring(File.channel(c, PixType), dtype=np.float32) for c in 'RGB']
        r = np.reshape(rgba[0], (Size[1], Size[0]))
        g = np.reshape(rgba[1], (Size[1], Size[0]))
        b = np.reshape(rgba[2], (Size[1], Size[0]))
        # alpha = np.reshape(rgb[3], (Size[1], Size[0]))
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

def convert_visuals_to_numpy(visuals):
    for key, t in visuals.items():
        t = t.permute(1,2,0)
        tmp = np.squeeze(t.detach().cpu().numpy())
        visuals[key] = tmp
    return visuals


def print_current_errors(epoch, i, errors, t):
    message = '(epoch: %d, iters: %d, time: %.3f)' % (epoch, i, t)
    for k, v in errors.items():
        v = v.mean().float()
        message += '%s: %.3f ' % (k, v)
    print(message)

def save_current_images(visuals, epoch, step):
    ## convert tensors to numpy arrays
    visuals = convert_visuals_to_numpy(visuals)

    # if self.use_html:  # save images to a html file
    for label, image_numpy in visuals.items():
        img_path = './summary/' + 'epoch%.3d_iter%.3d_%s.png' % (epoch, step, label)
        if label == 'input':
            print(label, image_numpy.sum(axis=(0, 1)), np.max(image_numpy))
            tonemapping(image_numpy, img_path, gamma=2.4, percentile=99, max_mapping=0.8)
        elif label == 'im':
            hdr = image_numpy * 255.0
            hdr = Image.fromarray(hdr.astype('uint8'))
            hdr.save(img_path)
        else:
            print(label, image_numpy.sum(axis=(0, 1)), np.max(image_numpy))
            tonemapping(image_numpy, img_path)
            # if label == 'warped' or 'fake_image':


def save_test_images(visuals, nm):
    ## convert tensors to numpy arrays
    visuals = convert_visuals_to_numpy(visuals)

    # if self.use_html:  # save images to a html file
    for label, image_numpy in visuals.items():

        if label == 'fake_image':
            im_path = './results/' + nm + '_' + label + '.jpg'
            tonemapping(image_numpy, im_path)

            hdr_path = './results/' + nm + '_' + label + '.exr'
            write_exr(hdr_path, image_numpy)

            # print (image_numpy.sum(axis=(0, 1)))

        if label == 'warped':
            im_path = './results/' + nm + '_' + label + '.jpg'
            tonemapping(image_numpy, im_path)

            hdr_path = './results/' + nm + '_' + label + '.exr'
            write_exr(hdr_path, image_numpy)

            # print (image_numpy.sum(axis=(0, 1)))