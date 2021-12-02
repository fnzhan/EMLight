import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import data
from torch.optim import lr_scheduler
import numpy as np
import pickle
from util import PanoramaHandler, TonemapHDR, tonemapping
from PIL import Image
import util
import DenseNet
from gmloss import SamplesLoss

import imageio
imageio.plugins.freeimage.download()

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
h = PanoramaHandler()
batch_size = 1

save_dir = "./checkpoints"
test_dir = '/home/fangneng.zfn/datasets/LavalIndoor/test/'
hdr_train_dataset = data.ParameterDataset(train_dir)
dataloader = DataLoader(hdr_train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Model = DenseNet.DenseNet().to(device)
load_weight = True
if load_weight:
    Model.load_state_dict(torch.load("./checkpoints/latest_net.pth"))
    print ('load trained model')
tone = util.TonemapHDR(gamma=2.4, percentile=99, max_mapping=0.9)

# torch.set_grad_enabled(True)
# Model.test()
for i, para in enumerate(dataloader):
    if i >= 100:
        break
    ln = 42

    nm = para['name'][0]

    input = para['crop'].to(device)
    pred = Model(input)

    distribution_pred = pred['distribution']
    distribution_gt = para['distribution'].to(device)

    rgb_ratio_pred = pred['rgb_ratio']
    rgb_ratio_gt = para['rgb_ratio'].to(device)

    intensity_pred = pred['intensity'] * 500
    intensity_gt = para['intensity'].to(device) * 500

    print (intensity_pred, intensity_gt)

    # save images
    dirs = util.polyhedron(1)
    dirs = torch.from_numpy(dirs).float()
    dirs = dirs.view(1, ln*3).cuda()

    size = torch.ones((1, ln)).cuda() * 0.0025
    # print (rgb_ratio_gt[:1].shape)
    rgb_ratio_gt = rgb_ratio_gt[0].view(1, 1, 3).repeat(1, 42, 1)
    intensity_gt = intensity_gt[0].view(1, 1, 1).repeat(1, 42, 1)
    distribution_gt = distribution_gt[0].view(1, 42, 1)
    color_gt = rgb_ratio_gt * intensity_gt * distribution_gt
    color_gt = color_gt.view(1, ln*3)

    # hdr = util.convert_to_panorama(dirs, size, color_pred)
    # hdr = np.squeeze(hdr[0].detach().cpu().numpy())

    # env = tone(hdr) * 255.0
    # env = np.transpose(env, (1,2,0))
    # env = Image.fromarray(env.astype('uint8'))
    # env.save('./summary/{}_env.jpg'.format(i))

    rgb_ratio = np.squeeze(rgb_ratio_pred[0].view(3).detach().cpu().numpy())
    intensity = np.squeeze(intensity_pred[0].detach().cpu().numpy())
    distribution = np.squeeze(distribution_pred[0].view(ln).detach().cpu().numpy())
    parametric_lights = {"distribution": distribution, "rgb_ratio": rgb_ratio, 'intensity': intensity}

    with open('./results/' + nm + '.pickle', 'wb') as handle:
        pickle.dump(parametric_lights, handle, protocol=pickle.HIGHEST_PROTOCOL)


    # rgb_ratio_pred = rgb_ratio_pred[0].view(1, 1, 3).repeat(1, 42, 1)
    # intensity_pred = intensity_pred[0].view(1, 1, 1).repeat(1, 42, 1)
    # distribution_pred = distribution_pred[0].view(1, 42, 1)
    # color_pred = rgb_ratio_pred * intensity_pred * distribution_pred  # 1, 42, 1    1, 42, 3
    # color_pred = color_pred.view(1, ln * 3)
    #
    # hdr_pred = util.convert_to_panorama(dirs, size, color_pred)
    # hdr_pred = np.squeeze(hdr_pred[0].detach().cpu().numpy())
    # hdr_pred = np.transpose(hdr_pred, (1, 2, 0))
    #
    # hdr_pred = hdr_pred.astype('float32')
    # hdr_path = './results/{}.exr'.format(nm)
    # util.write_exr(hdr_path, hdr_pred)
    #
    # ldr_pred = tone(hdr_pred) * 255.0
    # ldr_pred = Image.fromarray(ldr_pred.astype('uint8'))
    # ldr_pred.save('./results/{}.jpg'.format(nm))

    print (i)