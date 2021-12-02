"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import os
import importlib
import torch.utils.data
import numpy as np
import pickle
import cv2
from PIL import Image
import util


class LavalIndoorDataset():

    def __init__(self, opt):
        self.opt = opt
        self.pairs = self.get_paths(opt)

        size = len(self.pairs)
        self.dataset_size = size

        self.tone = util.TonemapHDR(gamma=2.4, percentile=50, max_mapping=0.5)

        h, w = 128, 256
        steradian = np.linspace(0, h, num=h, endpoint=False) + 0.5
        steradian = np.sin(steradian / h * np.pi)
        steradian = np.tile(steradian.transpose(), (w, 1))
        steradian = steradian.transpose()
        self.steradian = steradian[..., np.newaxis]


    def is_image_file(self, filename):
        IMG_EXTENSIONS = [
            '.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG',
            '.ppm', '.PPM', '.bmp', '.BMP', '.tiff', '.webp', '.exr']
        return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

    def get_paths(self, opt):
        # if opt.phase == 'train':
        dir = 'pkl/'
        pkl_dir = opt.dataroot + dir
        pairs = []
        nms = os.listdir(pkl_dir)

        for nm in nms:
            if nm.endswith('.pickle'):
                pkl_path = pkl_dir + nm
                warped_path = pkl_path.replace(dir, 'warped/')
                # warped_path = pkl_path.replace(dir, 'test/')
                warped_path = warped_path.replace('pickle', 'exr')
                # print (warped_path)
                if os.path.exists(warped_path):
                    pairs.append([pkl_path, warped_path])
        return pairs

    def __getitem__(self, index):

        ln = 128
        # read .exr image
        pkl_path, warped_path = self.pairs[index]

        handle = open(pkl_path, 'rb')
        pkl = pickle.load(handle)

        crop_path = warped_path.replace('warped', 'crop')
        crop = util.load_exr(crop_path)
        crop, alpha = self.tone(crop)
        crop = cv2.resize(crop, (128, 128))
        crop = torch.from_numpy(crop).float().cuda().permute(2, 0, 1)

        hdr = util.load_exr(warped_path)

        hdr_intensity = 0.3 * hdr[..., 0] + 0.59 * hdr[..., 1] + 0.11 * hdr[..., 2]
        max_intensity_ind = np.unravel_index(np.argmax(hdr_intensity, axis=None), hdr_intensity.shape)
        max_intensity = hdr_intensity[max_intensity_ind]
        map = hdr_intensity > (max_intensity * 0.05)
        map = np.expand_dims(map, axis=0)
        map = np.array(map).astype('uint8')
        map = torch.from_numpy(map).float()
        warped = np.transpose(hdr, (2, 0, 1))
        warped = torch.from_numpy(warped)
        warped = warped * alpha

        dist_gt = torch.from_numpy(pkl['distribution']).float().cuda()
        intensity_gt = torch.from_numpy(np.array(pkl['intensity'])).float().cuda() * 0.01
        rgb_ratio_gt = torch.from_numpy(np.array(pkl['rgb_ratio'])).float().cuda()
        ambient_gt = torch.from_numpy(pkl['ambient']).float().cuda() / (128 * 256)

        intensity_gt = intensity_gt.view(1, 1, 1).repeat(1, ln, 3)
        dist_gt = dist_gt.view(1, ln, 1).repeat(1, 1, 3)
        rgb_ratio_gt = rgb_ratio_gt.view(1, 1, 3).repeat(1, ln, 1)

        dirs = util.sphere_points(ln)
        dirs = torch.from_numpy(dirs).float().view(1, ln * 3).cuda()
        size = torch.ones((1, ln)).cuda().float() * 0.0025
        light_gt = (dist_gt * intensity_gt * rgb_ratio_gt).view(1, ln * 3)
        env_gt = util.convert_to_panorama(dirs, size, light_gt)
        ambient_gt = ambient_gt.view(3, 1, 1).repeat(1, 128, 256).cuda()
        env_gt = env_gt.view(3, 128, 256) + ambient_gt
        env_gt = env_gt * alpha

        input_dict = {'input': env_gt, 'crop': crop, 'warped': warped, 'map': map,
                      'distribution': dist_gt, 'intensity': intensity_gt,
                      'name': pkl_path.split('/')[-1].split('.')[0]}

        return input_dict



    def __len__(self):
        return self.dataset_size


def create_dataloader(opt):
    dataset = LavalIndoorDataset(opt)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batchSize,
        shuffle=not opt.serial_batches,
        num_workers=int(opt.nThreads),
        drop_last=opt.isTrain
    )
    return dataloader
