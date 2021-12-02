from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image
import util
import os

tm = util.TonemapHDR(gamma=1., percentile=90, max_mapping=0.8)


def Normalize(dir):
    dir_l = torch.sqrt(torch.sum(dir ** 2, dim=1) + 1e-6)
    return dir / dir_l[:, None]


def tonemapping(im, sv_path):
    power_im = np.power(im, 1 / 2.2)
    # print (np.amax(power_im))
    non_zero = power_im > 0
    if non_zero.any():
        r_percentile = np.percentile(power_im[non_zero], 90)
    else:
        r_percentile = np.percentile(power_im, 90)
    alpha = 0.8 / (r_percentile + 1e-10)
    tonemapped_im = np.multiply(alpha, power_im)

    tonemapped_im = np.clip(tonemapped_im, 0, 1)
    hdr = tonemapped_im * 255.0
    hdr = Image.fromarray(hdr.astype('uint8'))
    hdr.save(sv_path)


def check_grad(name):
    def hook(grad):
        # grads[name] = grad
        print("{}_grad".format(name), grad[-1])
        # if name == "sizes":
        #     return 0.1 * grad
        # if name == "colors":
        #     return 0.1 * grad

    return hook


def check_module(name):
    def hookFunc(module, gradInput, gradOutput):
        print("name", name)
        n = torch.norm(gradInput[-1], 2, 1)
        print(n.min().item(), n.max().item())
        # for v in gradInput:
        #     print(v)
        # for v in gradOutput:
        #     print(v)
        # return (gradInput[0] * 1, gradInput[1]* 0)

    return hookFunc


class Panorama(nn.Module):
    def __init__(self, N=3, latitude=128):
        super(Panorama, self).__init__()
        self.N = N
        # phi (dim=1) is in [0, 2pi] and theta (dim=0) is in [0, pi]
        grid_latitude, grid_longitude = torch.meshgrid(
            [torch.arange(latitude, dtype=torch.float), torch.arange(2 * latitude, dtype=torch.float)])
        grid_latitude = grid_latitude.add(0.5)
        grid_longitude = grid_longitude.add(0.5)
        grid_latitude = grid_latitude.mul(np.pi / latitude)
        grid_longitude = grid_longitude.mul(np.pi / latitude)

        x = torch.sin(grid_latitude) * torch.cos(grid_longitude)
        y = torch.sin(grid_latitude) * torch.sin(grid_longitude)
        z = torch.cos(grid_latitude)
        self.register_buffer('xyz', torch.stack((x, y, z)))

        self.fc = nn.Linear(3072, 512)  # , bias=False
        # self.fc2 = nn.Linear(512, 512)

        self.bn_3072 = nn.BatchNorm1d(3072)
        self.bn_512 = nn.BatchNorm1d(512)

        self.fc_dir = nn.Linear(512, N * 3)

        # nn.init.zeros_(self.fc_dir.weight)
        #
        # representation = -1.0 * torch.Tensor([-0.0092,  0.9600,  0.2798,
        #                            -0.7550, -0.4581, 0.4692,
        #                            0.4283, -0.6480, 0.6298])
        #
        # self.fc_dir.bias = nn.Parameter(representation)

        self.fc_size = nn.Linear(512, N * 1)
        # self.fc_size.register_backward_hook(check_module('sizes'))

        self.fc_color = nn.Linear(512, N * 3)
        # self.fc_color.register_backward_hook(check_module('colors'))

        self.fc_ambient = nn.Linear(512, 3)

        self.elu = nn.ELU()
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.softplus = nn.Softplus(threshold=100)

    def forward(self, input, epoch=0, batch=0):
        # input = self.bn_3072(input)
        input = self.fc(input)
        # input = self.bn_512(input)
        input = self.elu(input)

        dirs = self.tanh(self.fc_dir(input)).view(-1, self.N, 3)  # assume dir in range [-1, 1] and normalized
        # dirs = self.fc_dir(input).view(-1, 3)
        # dirs = Normalize(dirs)
        norms = torch.norm(dirs, 2, 2, keepdim=True)
        dirs = dirs.div(norms)

        dirs = dirs.view(-1, self.N * 3)
        sizes = torch.sigmoid(self.fc_size(input)).mul(np.radians(60)).add(0.02)
        # # assume size in [0, 1]  which is normalized by 4*pi
        if torch.is_grad_enabled() and epoch % 2 == 0 and batch == 0:
            sizes.register_hook(check_grad('sizes'))

        colors = self.sigmoid(self.fc_color(input)).mul(100)
        if torch.is_grad_enabled() and epoch % 2 == 0 and batch == 0:
            colors.register_hook(check_grad('colors'))

        ambient = self.sigmoid(self.fc_ambient(input)).mul(10.)  # assume ambient in range [0, 1]

        lights = self.convert_to_panorama(dirs, sizes, colors)
        # lights = self.convert_to_panorama_log(dirs, sizes, colors)
        return lights, ambient, dirs, sizes, colors  # , torch.sum(dirs.view(-1, 3) * dirs.view(-1, 3), dim=1)

    def convert_to_panorama(self, dirs, sizes, colors):
        nbatch = dirs.shape[0]
        lights = torch.zeros((nbatch, 3, self.xyz.shape[1], self.xyz.shape[2]), dtype=dirs.dtype, device=dirs.device)
        nlights = self.N
        for i in range(nlights):
            lights = lights + (colors[:, 3 * i + 0:3 * i + 3][:, :, None, None]) * (
                torch.exp(
                    (torch.matmul(dirs[:, 3 * i + 0:3 * i + 3], self.xyz.view(3, -1)).
                     view(-1, self.xyz.shape[1], self.xyz.shape[2]) - 1) /
                    (sizes[:, i]).view(-1, 1, 1))[:, None, :, :])
        return lights

    def convert_to_panorama_log(self, dirs, sizes, colors):
        nbatch = dirs.shape[0]
        lights = torch.zeros((nbatch, 3, self.xyz.shape[1], self.xyz.shape[2]), dtype=dirs.dtype, device=dirs.device)
        nlights = self.N
        for i in range(nlights):
            lights = lights + colors[:, 3 * i + 0:3 * i + 3][:, :, None, None] + (
                    torch.matmul(dirs[:, 3 * i + 0:3 * i + 3], self.xyz.view(3, -1)).
                    view(-1, self.xyz.shape[1], self.xyz.shape[2]) - 1) * (sizes[:, i]).view(-1, 1, 1)
        # lights = torch.clamp(lights, -5.0, 10000)
        return lights


class LightParamter(nn.Module):
    def __init__(self, N=3, latitude=128):
        super(LightParamter, self).__init__()
        self.N = N

        grid_latitude, grid_longitude = torch.meshgrid(
            [torch.arange(latitude, dtype=torch.float), torch.arange(2 * latitude, dtype=torch.float)])
        grid_latitude = grid_latitude.add(0.5)
        grid_longitude = grid_longitude.add(0.5)
        grid_latitude = grid_latitude.mul(np.pi / latitude)
        grid_longitude = grid_longitude.mul(np.pi / latitude)

        x = torch.sin(grid_latitude) * torch.cos(grid_longitude)
        y = torch.sin(grid_latitude) * torch.sin(grid_longitude)
        z = torch.cos(grid_latitude)
        self.register_buffer('xyz', torch.stack((x, y, z)))

        self.fc = nn.Linear(3072, 1024)  # , bias=False
        self.fc2 = nn.Linear(1024, 600)
        # self.fc2 = nn.Linear(512, 512)
        self.bn_3072 = nn.BatchNorm1d(3072)
        self.bn_512 = nn.BatchNorm1d(512)
        self.fc_dir = nn.Linear(512, N * 3)
        self.fc_size = nn.Linear(512, N * 1)
        self.fc_color = nn.Linear(512, N * 3)
        self.fc_ambient = nn.Linear(512, 3)
        self.elu = nn.ELU()
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.softplus = nn.Softplus(threshold=100)
        self.softmax = nn.Softmax(dim=2)
        self.conv = nn.Conv2d(3, 3, kernel_size=1, stride=1,padding=0)

    def forward(self, input):



        # output = nn.

        # mesh: 10 *20, color: 3, radius = 1,
        output = self.fc(input)
        output = self.fc2(output)
        # output = output.view(-1, 3, 200)
        output = self.sigmoid(output) * 10
        output = output.view(-1, 3, 10, 20)
        # print(output.shape)
        # output = self.conv(output)
        # output = self.relu(output)
        # output = output.view(-1, 3, 200, 1)

        # print (output.shape)
        # print (1/0)

        # color = output[...,:3]
        # size = output[...,3:]

        return {
            "color": output,
            # "size": size,
        }


    def convert_to_panorama(self, dirs, sizes, colors):
        nbatch = dirs.shape[0]
        lights = torch.zeros((nbatch, 3, self.xyz.shape[1], self.xyz.shape[2]), dtype=dirs.dtype, device=dirs.device)
        nlights = self.N
        for i in range(nlights):
            lights = lights + (colors[:, 3 * i + 0:3 * i + 3][:, :, None, None]) * (
                torch.exp(
                    (torch.matmul(dirs[:, 3 * i + 0:3 * i + 3], self.xyz.view(3, -1)).
                     view(-1, self.xyz.shape[1], self.xyz.shape[2]) - 1) /
                    (sizes[:, i]).view(-1, 1, 1))[:, None, :, :])
        return lights


def generate_steradian(height, width, multiply=True):
    steradian = np.linspace(0, height, num=height, endpoint=False) + 0.5
    steradian = np.sin(steradian / height * np.pi)
    steradian = np.tile(steradian.transpose(), (width, 1))
    steradian = steradian.transpose()
    if multiply:
        pixel_area = ((2 * np.pi) / width) * ((1 * np.pi) / height)
        steradian *= pixel_area
    return steradian.astype(np.float32)


def prepare_gt_panorama(hdr_img, threshold=None, shape=None):
    weight = generate_steradian(height=hdr_img.shape[0], width=hdr_img.shape[1])
    hdr_intensity = 0.2126 * hdr_img[:, :, 0] + 0.7152 * hdr_img[:, :, 1] + 0.0722 * hdr_img[:, :, 0]
    if threshold is None:
        threshold = hdr_intensity.max() / 20.

    mask = np.where(hdr_intensity < threshold)

    if mask[0].size != 0:
        ambient = np.sum(hdr_img[mask] * weight[mask][:, np.newaxis], axis=0, dtype=np.float32) / np.sum(weight[mask],
                                                                                                         dtype=np.float32)
    else:
        ambient = np.zeros([3], dtype=np.float32)
    hdr_img[mask] = 0.0
    if shape:
        if isinstance(shape, tuple) and len(shape) == 2:
            hdr_img = cv2.resize(hdr_img, shape, interpolation=cv2.INTER_LINEAR)
        elif isinstance(shape, int):
            hdr_img = cv2.resize(hdr_img, (2 * shape, shape), interpolation=cv2.INTER_LINEAR)
    return hdr_img, ambient


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Panorama(N=3, latitude=128)
    model = model.to(device)

    lr_base = 0.0001
    betas = (0.5, 0.9)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_base, betas=betas)
    # filter_keys = ["fc_dir.weight", "fc_size.weight", "fc_color.weight",
    #                "fc_dir.bias", "fc_size.bias", "fc_color.bias"]
    #
    # dir_params = list(filter(lambda kv: "fc_dir" in kv[0], model.named_parameters()))
    # size_params = list(filter(lambda kv: "fc_size" in kv[0], model.named_parameters()))
    # color_params = list(filter(lambda kv: "fc_color" in kv[0], model.named_parameters()))
    # rest_params = list(filter(lambda kv: kv[0] not in filter_keys, model.named_parameters()))
    #
    # optimizer = torch.optim.Adam(
    #     [
    #         {"params": [i[1]for i in dir_params], "lr": lr_base * 0.01},
    #         {"params": [i[1]for i in size_params], "lr": lr_base * 10},
    #         {"params": [i[1]for i in color_params], "lr": lr_base * 1},
    #         {"params": [i[1] for i in rest_params]},
    #     ],
    #     lr=lr_base, betas=betas,
    # )

    # optimizer = optim.SGD(model.parameters(), lr=lr_3d, momentum=0.9)

    # lr_decay_iters = 10000
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=lr_decay_iters, gamma=0.1)

    inputs = torch.randn((1, 3072)).to(device)
    if os.path.exists("./intermedia.npy"):
        inputs = np.expand_dims(np.load("./intermedia.npy"), 0)
        inputs = torch.from_numpy(inputs).to(device)
    # hdr_path = "/home/changgongzhang/Dukto/9C4A0006-5133111e97.hdr"
    # hdr_img = cv2.imread(hdr_path, flags=cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)[:, :, ::-1]

    # hdr_path = "/home/changgongzhang/PycharmProjects/efficient_densenet_pytorch/panorama/pre_process_training/9C4A0531-176ffd54e6_44.exr"
    # hdr_path = "/home/changgongzhang/PycharmProjects/efficient_densenet_pytorch/panorama/pre_process_training/9C4A0003-e05009bcad_1.exr"
    hdr_path = "./pre_process_training/9C4A0004-db1d4a14f3_0.exr"
    h = util.PanoramaHandler()
    hdr_img = h.read_exr(hdr_path)

    # target = (hdr_img, np.array([0.20155587, 0.17011154, 0.12357724], dtype=np.float32))
    # target = prepare_gt_panorama(hdr_img, threshold=None, shape=256)
    target = h.prepare_gt_panorama(hdr_img, threshold=0.0)
    to_tensor = torchvision.transforms.ToTensor()
    hdr = to_tensor(target[0])
    # ambient = torch.from_numpy(target[1])
    ambient = torch.from_numpy(
        np.array([0.001990176271647215, 0.0009630646673031151, 0.00016221970145124942], dtype=np.float32))

    hdr_gt = hdr
    hdr_gt = hdr_gt.unsqueeze(0).to(device)
    ambient = ambient.unsqueeze(0).to(device)
    l2_loss = nn.MSELoss().to(device)
    l1_loss = nn.L1Loss().to(device)
    model.train()

    weights = torch.from_numpy(generate_steradian(128, 2 * 128, True)).to(device)
    weights = 128 * 256 * weights[None, None, :, :] / (4 * np.pi)
    for epoch in range(10000 * 100):
        optimizer.zero_grad()
        outputs = model(inputs, epoch)

        hdr_pred = outputs[0]

        # hdr_loss = l2_loss(weights * hdr_pred, weights * hdr_gt)
        hdr_loss = l2_loss(hdr_pred, hdr_gt)
        ambient_loss = l2_loss(outputs[1], ambient)
        error = 20 * hdr_loss + ambient_loss
        print(epoch, "hdr_loss: ", hdr_loss.item())
        if epoch == 0 or (epoch + 1) % 2000 == 0:
            print("hdr_loss:", hdr_loss.item(), " ambient_loss:", ambient_loss.item(), " max value: ",
                  outputs[0].max().item())
            print("dirs: ", outputs[2].detach().cpu().numpy().reshape(-1, 3))
            print("size:", outputs[3].detach().cpu().numpy().reshape(-1, 3))
            print("color: ", outputs[4].detach().cpu().numpy().reshape(-1, 3))
            alpha, gt_ldr = tm(np.transpose(hdr.cpu().numpy(), (1, 2, 0)), clip=True)
            # print(alpha)
            gt_ldr = (gt_ldr * 255.0).astype(np.uint8)

            pred_hdr = np.transpose(np.squeeze(hdr_pred.detach().cpu().numpy()), (1, 2, 0))
            # pred_ambient = np.squeeze(pred[1][0].cpu().numpy())
            # pred_hdr = pred_hdr + pred_ambient[np.newaxis, np.newaxis, :]
            alpha, pred_ldr = tm(pred_hdr, clip=True, alpha=alpha)  # , alpha=alpha
            pred_ldr = (pred_ldr * 255.0).astype(np.uint8)

            cmp = np.vstack((gt_ldr, pred_ldr))
            cmp = Image.fromarray(cmp)
            cmp.save(
                os.path.join("/home/changgongzhang/PycharmProjects/efficient_densenet_pytorch/panorama/results",
                             "{}.jpg".format(epoch)))

        error.backward()
        optimizer.step()

    # # inputs = torch.zeros((1, 3 * 8 + 3))

    # lighting_sources = np.array([[
    #     -0.94782092, -0.00174248, 0.31879848, 1.0, 0.02141743396088052, 100.43005105, 111.44961806, 126.33202083,
    #     0.97524598, -0.19746356, -0.09951595, 1.0, 0.005262753039722748, 43.50644635, 23.90164926, 14.95728657,
    #     -0.62688431, 0.20015295, -0.75296405, 1.0, 0.11169028538020485, 15.64642994, 13.07596046, 9.37460972,
    #     0.20155587, 0.17011154, 0.12357724]], dtype=np.float32)
    # lighting_sources = np.repeat(lighting_sources, 1, axis=0)

    # debug convert_to_panorama
    # dirs = np.array(
    #     [[-0.94782092, -0.00174248, 0.31879848, 0.97524598, -0.19746356, -0.09951595, -0.62688431, 0.20015295,
    #       -0.75296405]], dtype=np.float32)
    # dirs = dirs.repeat(2, axis=0)
    #
    # sizes = np.array([[0.02141743396088052, 0.005262753039722748, 0.11169028538020485]], dtype=np.float32) / (4 * np.pi)
    # sizes = sizes.repeat(2, axis=0)
    #
    # colors = np.array([[100.43005105, 111.44961806, 126.33202083, 43.50644635, 23.90164926, 14.95728657, 15.64642994,
    #                     13.07596046, 9.37460972]], dtype=np.float32)
    # colors = colors.repeat(2, axis=0)
    # ambient = np.array([0.20155587, 0.17011154, 0.12357724], dtype=np.float32)

    # inputs = torch.from_numpy(lighting_sources)
    # inputs = inputs.to(device)
    #
    # hdr_path = "/home/changgongzhang/Dukto/0114_rec.hdr"
    # hdr_img = cv2.imread(hdr_path, flags=cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)[:, :, ::-1]
    # target = prepare_gt_panorama(hdr_img, shape=600)
    # to_tensor = torchvision.transforms.ToTensor()
    # hdr = to_tensor(target[0])
    # ambient = torch.from_numpy(target[1])
    # hdr = hdr.unsqueeze(0)
    # ambient = ambient.unsqueeze(0)
    #
    # loss = nn.MSELoss()
    #
    # model = Panorama(N=3, latitude=600)
    # lights = model.convert_to_panorama(torch.from_numpy(dirs), torch.from_numpy(sizes), torch.from_numpy(colors))
    #
    # lights = torch.squeeze(lights[-1]).cpu().numpy()
    # lights = np.transpose(lights, (1, 2, 0))
    # outputs = model(inputs)
    # panorama = lights + ambient[np.newaxis, np.newaxis, :]
    # print(loss(lights, hdr).item())
    # print(torch.sum(ret[0] - ret[1]))

    # ret = torch.squeeze(ret[1]).cpu().numpy()
    # ret = np.transpose(ret, (1, 2, 0))
    # print(ret.shape, ret.dtype)
    #
    # panorama = lights[..., ::-1].astype(np.float32)
    # hdr_path = "/home/changgongzhang/Dukto/0114_rec_torch.hdr"
    # cv2.imwrite(hdr_path, panorama)

    # hdr_path = "/home/changgongzhang/Dukto/0114.hdr"
    # hdr_img = cv2.imread(hdr_path, flags=cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)[:, :, ::-1]
    # print(np.sum(hdr_img - ret))
    # loss = F.nll_loss(output[0], target)
    # loss = F.nll_loss(output[1], target)
