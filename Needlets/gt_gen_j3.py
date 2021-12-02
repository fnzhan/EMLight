import os
from PIL import Image
import numpy as np
import torch
from sphere_needlets import spneedlet_eval, SNvertex, visualize
from utils import tonemapping, getSolidAngleMap
import utils

jmax = 3
B = 2.0
h, w = 128, 256
bs_dir = '/home/fangneng.zfn/datasets/LavalIndoor/'
exr_dir = bs_dir + 'marc/warpedHDROutputs/'
crop_dir = bs_dir + 'nips/exr/'
sv_dir = bs_dir + 'nips/needlets_j3/'
SN_Matrix = np.load('SN_Matrix3.npy')  # 128*256, 31

_, nCoeffs = SN_Matrix.shape
solidAngles = getSolidAngleMap(w).reshape((-1))

handle = utils.PanoramaHandler()
tone = utils.TonemapHDR(gamma=2.4, percentile=50, max_mapping=0.5)

nms = os.listdir(exr_dir)
nms = nms[:1000]
idx = 0
for nm in nms:
    if nm.endswith('.exr'):
        exr_path = exr_dir + nm
        exr = handle.read_exr(exr_path)
        exr = handle.resize_panorama(exr, (w, h))
        exr = np.array(exr).reshape((-1, 3))

        crop_path = crop_dir + nm
        crop = handle.read_hdr(crop_path)
        _, alpha = tone(crop)
        exr = exr * alpha

        SN_Coeffs = np.zeros((nCoeffs, 3))
        for i in range(nCoeffs):
            SN_Coeffs[i, 0] = np.sum(exr[:, 0] * SN_Matrix[:, i] * solidAngles)
            SN_Coeffs[i, 1] = np.sum(exr[:, 1] * SN_Matrix[:, i] * solidAngles)
            SN_Coeffs[i, 2] = np.sum(exr[:, 2] * SN_Matrix[:, i] * solidAngles)

        np.save(sv_dir+nm.replace('exr', 'npy'), SN_Coeffs)


        # rec = np.dot(SN_Matrix, SN_Coeffs)
        # rec = rec.reshape((h, w, 3))
        # exr = exr.reshape((h, w, 3))
        #
        # exr_energy = exr[:, :, 0] * 0.3 + exr[:, :, 1] * 0.59 + exr[:, :, 2] * 0.11
        # rec_energy = rec[:, :, 0] * 0.3 + rec[:, :, 1] * 0.59 + rec[:, :, 2] * 0.11
        # print (exr_energy.sum(), rec_energy.sum())
        #
        # im_sv_dir = bs_dir + 'nips/tmp/'
        # rec = tonemapping(rec) * 255.0
        # rec = Image.fromarray((rec).astype('uint8'))
        # rec.save(im_sv_dir + nm.replace('.exr', '_rec.jpg'))
        #
        # im = tonemapping(exr) * 255.0
        # im = Image.fromarray((im).astype('uint8'))
        # im.save(im_sv_dir + nm.replace('exr', 'jpg'))

        idx += 1
        print (idx)
