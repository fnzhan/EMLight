import cv2
import numpy as np
from scipy import ndimage
from PIL import Image
import pickle
import os.path
import imageio
import time
import vtk
from vtk.util import numpy_support

import detect_util
import util

imageio.plugins.freeimage.download()
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


def rgb_to_intenisty(rgb):
    intensity = 0.3 * rgb[..., 0] + 0.59 * rgb[..., 1] + 0.11 * rgb[..., 2]
    return intensity

def polar_to_cartesian(phi_theta):
    phi, theta = phi_theta
    # sin(theta) * cos(phi)
    x = np.sin(theta) * np.cos(phi)
    # sin(theta) * sin(phi)
    y = np.sin(theta) * np.sin(phi)
    # cos(theta)
    z = np.cos(theta)
    return np.array([x, y, z])

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


def extract_light_info(hdr):
    h, w, _ = hdr.shape

    info = {"distribution": None, "rgb_ratio": None, "intensity": None}
    steradian = np.linspace(0, h, num=h, endpoint=False) + 0.5
    steradian = np.sin(steradian / h * np.pi)
    steradian = np.tile(steradian.transpose(), (w, 1))
    steradian = steradian.transpose()
    # pixel_area = ((2 * np.pi) / w) * ((1 * np.pi) / h)
    # steradian *= pixel_area
    weight_hdr = steradian[..., np.newaxis] * hdr

    anchors = polyhedron(1)
    ln, _ = anchors.shape

    # distribution = np.zeros(ln)
    # rgb_ratio = np.zeros((ln, 3))
    # intensity = 0
    rgbs = np.zeros((ln, 3))

    for i in range(h):
        for j in range(w):
            color = weight_hdr[i, j]
            polar = [j/w * 2*np.pi, i/h * np.pi] #phi_theta
            coord = polar_to_cartesian(polar)

            idx = 0
            distance = 3
            for n in range(ln):
                anchor = anchors[n]
                tmp = np.linalg.norm(coord - anchor)
                if tmp < distance:
                    idx = n
                    distance = tmp
            rgbs[idx, :] += color

    rgbs = rgbs + 1e-9  # (w*h/ln) #42, 3
    tmp = np.sum(rgbs, axis=0)

    rgb_ratio = tmp / np.sum(tmp)
    tmp1 = 0.3*rgb_ratio[0] + 0.59*rgb_ratio[1] + 0.11*rgb_ratio[2]

    total_energy = 0.3*tmp[0] + 0.59*tmp[1] + 0.11*tmp[2]
    anchors_energy = 0.3*rgbs[:, 0] + 0.59*rgbs[:, 1] + 0.11*rgbs[:, 2]
    distribution = anchors_energy / total_energy

    intensity = total_energy / tmp1

    parametric_lights = {"distribution": distribution, "rgb_ratio": rgb_ratio, 'intensity':intensity}
    return parametric_lights

# train_dir = '/home/fangneng.zfn/datasets/LavalIndoor/nips/'
bs_dir = '/home/fangneng.zfn/datasets/LavalIndoor/'
# hdr_dir = bs_dir + 'marc/warpedHDROutputs/'

sv_dir = bs_dir + 'nips/pkl_masked/'
pkl_dir = bs_dir + 'nips/pkl_tmp/'
nms = os.listdir(pkl_dir)
# nms = nms[6475:]

i = 0
for nm in nms:
    if nm.endswith('.pickle'):
        pkl_path = pkl_dir + nm
        handle = open(pkl_path, 'rb')
        pkl = pickle.load(handle)

        intensity = pkl['intensity']
        pkl['intensity'] = intensity

        with open((sv_dir + nm), 'wb') as handle2:
            pickle.dump(pkl, handle2, protocol=pickle.HIGHEST_PROTOCOL)
        i += 1
        print (i)