import os
from PIL import Image
import numpy as np
import torch
from sphere_needlets import spneedlet_eval, SNvertex, visualize
from utils import tonemapping, getSolidAngleMap
import utils

jmax = 2
B = 2.0
h, w = 128, 256
bs_dir = '/home/fangneng.zfn/datasets/LavalIndoor/'
exr_dir = bs_dir + 'marc/warpedHDROutputs/'
crop_dir = bs_dir + 'nips/exr/'
sv_dir = bs_dir + 'nips/needlets_j2_sparse_alpha/'
SN_Matrix = np.load('SN_Matrix2.npy')  # 128*256, 31

_, nCoeffs = SN_Matrix.shape
solidAngles = getSolidAngleMap(w).reshape((-1))

handle = utils.PanoramaHandler()
tone = utils.TonemapHDR(gamma=2.4, percentile=50, max_mapping=0.5)



nms = os.listdir(exr_dir)
# nms = nms[:1000]
idx = 0
for nm in nms:
    if nm.endswith('.exr'):
        exr_path = exr_dir + nm
        exr = handle.read_exr(exr_path)
        exr = handle.resize_panorama(exr, (w, h))
        exr = np.array(exr).reshape((-1, 3))

        crop_path = crop_dir + nm
        crop = handle.read_hdr(crop_path)
        _, alpha = tone(crop, gamma=False)
        exr = exr * alpha

        SN_Coeffs = np.zeros((nCoeffs, 3))
        for i in range(nCoeffs):
            SN_Coeffs[i, 0] = np.sum(exr[:, 0] * SN_Matrix[:, i] * solidAngles)
            SN_Coeffs[i, 1] = np.sum(exr[:, 1] * SN_Matrix[:, i] * solidAngles)
            SN_Coeffs[i, 2] = np.sum(exr[:, 2] * SN_Matrix[:, i] * solidAngles)

        j2 = SN_Coeffs[61:253, :]
        energy = (abs(j2)).sum(axis=1)
        thre = np.percentile(energy, 75)
        mask = (energy > thre)
        # print(np.array(mask).sum(), 192)
        SN_Coeffs[61:253, :] = j2 * mask[:, np.newaxis]

        j1 = SN_Coeffs[13:61, :]
        energy = (abs(j1)).sum(axis=1)
        thre = np.percentile(energy, 45)
        mask = (energy > thre)
        # print(np.array(mask).sum(), 48)
        SN_Coeffs[13:61, :] = j1 * mask[:, np.newaxis]

        j0 = SN_Coeffs[1:13, :]
        energy = (abs(j0)).sum(axis=1)
        thre = np.percentile(energy, 30)
        mask = (energy > thre)
        # print(np.array(mask).sum(), 12)
        SN_Coeffs[1:13, :] = j0 * mask[:, np.newaxis]


        if SN_Coeffs.max()>1000:
            print (nm)
            print (1/0)
        np.save(sv_dir+nm.replace('exr', 'npy'), SN_Coeffs)

        print (abs(SN_Coeffs).min(), abs(SN_Coeffs).max())

        idx += 1
        print (idx)
