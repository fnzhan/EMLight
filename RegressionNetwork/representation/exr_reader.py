import cv2
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from PIL import Image
import pickle
import time
import os
import imageio
imageio.plugins.freeimage.download()
import util

handle = util.PanoramaHandler()
tone = util.TonemapHDR()

bs_dir = '/home/fangneng.zfn/datasets/LavalIndoor/'
exr_dir = bs_dir + 'marc/warpedHDROutputs/'
sv_dir = bs_dir + 'marc/warpedim/'
nms = os.listdir(exr_dir)
# nms = nms[:100]

i = 0
for nm in nms:
    if nm.endswith('.exr'):
        exr_path = exr_dir + nm
        exr = handle.read_exr(exr_path)
        im = tone(exr, True)

        im = Image.fromarray((im*255.0).astype('uint8'))
        sv_path = sv_dir + nm.replace('exr', 'jpg')
        im.save(sv_path)
        i += 1
        print (i)