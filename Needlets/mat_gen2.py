from PIL import Image
import numpy as np
import torch
from sphere_needlets import spneedlet_eval, SNvertex, visualize
from utils import tonemapping, getSolidAngleMap
import utils
import imageio
imageio.plugins.freeimage.download()

handle = utils.PanoramaHandler()

jmax = 3
B = 2.0
h, w = 128, 256
nm = 'test2.exr'
im = handle.read_exr(nm)
im = handle.resize_panorama(im, (w, h))
im = np.array(im)

im = im.reshape((-1, 3))

pix1 = np.linspace(0, 1, h) * np.pi
pix2 = np.linspace(0, 2, w) * np.pi
X, Y = np.meshgrid(pix2, pix1)
X, Y = X.reshape(-1), Y.flatten().reshape(-1)

SN_Matrix = np.load('SN_Matrix3.npy')  # 128*256, 31
print (SN_Matrix.shape) # 1, 12, 48, 192 = 253, 768 = 256



_, nCoeffs = SN_Matrix.shape
SN_Coeffs = np.zeros((nCoeffs, 3))   # 31, 3
solidAngles = getSolidAngleMap(w).reshape((-1))
for i in range(nCoeffs):
    SN_Coeffs[i, 0] = np.sum(im[:, 0] * SN_Matrix[:, i] * solidAngles)
    SN_Coeffs[i, 1] = np.sum(im[:, 1] * SN_Matrix[:, i] * solidAngles)
    SN_Coeffs[i, 2] = np.sum(im[:, 2] * SN_Matrix[:, i] * solidAngles)

j3 = SN_Coeffs[253:, :]
mask = (abs(j3) > abs(j3).max()*0.1)
print (np.array(mask).sum(), 768*3)
SN_Coeffs[253:, :] = j3 * mask

j2 = SN_Coeffs[61:253, :]
mask = (abs(j2) > abs(j2).max()*0.1)
print (np.array(mask).sum(), 253*3)
SN_Coeffs[61:253, :] = j2 * mask



rec = np.dot(SN_Matrix, SN_Coeffs)
rec = rec.reshape((h, w, 3))
im = im.reshape((h, w, 3))

print (im.min(), np.percentile(im, 75), im.max())
print (rec.min(), np.percentile(rec, 75), rec.max())

im_energy = im[:, :, 0] * 0.3 + im[:, :, 1] * 0.59 + im[:, :, 2] * 0.11
rec_energy = rec[:, :, 0] * 0.3 + rec[:, :, 1] * 0.59 + rec[:, :, 2] * 0.11
print (im_energy.sum(), rec_energy.sum())

# hdr_path = './results/hdr/{}.hdr'.format(nm.split('.')[0])
imageio.imwrite('rec.hdr', rec.astype('float32'), format='hdr')

rec = tonemapping(rec) * 255.0
rec = Image.fromarray((rec).astype('uint8'))
rec.save('rec.jpg')

im = tonemapping(im) * 255.0
im = Image.fromarray((im).astype('uint8'))
im.save(nm.replace('exr', 'jpg'))
