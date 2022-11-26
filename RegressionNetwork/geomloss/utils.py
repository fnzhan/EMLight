# coding=utf-8
import torch
import numpy as np


def scal(α, f):
    # if batch:
    B = α.shape[0]
    return (α.view(B, -1) * f.view(B, -1)).sum(1)
    # else:
    #     return torch.dot(α.view(-1), f.view(-1))


class Sqrt0(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        result = input.sqrt()
        result[input < 0] = 0
        ctx.save_for_backward(result)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        result, = ctx.saved_tensors
        grad_input = grad_output / (2 * result)
        grad_input[result == 0] = 0
        return grad_input


def sqrt_0(x):
    return Sqrt0.apply(x)


def squared_distances(x, y):
    # if x.dim() == 2:
    #     D_xx = (x*x).sum(-1).unsqueeze(1)  # (N,1)
    #     D_xy = torch.matmul( x, y.permute(1,0) )  # (N,D) @ (D,M) = (N,M)
    #     D_yy = (y*y).sum(-1).unsqueeze(0)  # (1,M)
    # elif x.dim() == 3:  # Batch computation
    D_xx = (x * x).sum(-1).unsqueeze(2)  # (B,N,1)
    D_xy = torch.matmul(x, y.permute(0, 2, 1))  # (B,N,D) @ (B,D,M) = (B,N,M)
    D_yy = (y * y).sum(-1).unsqueeze(1)  # (B,1,M)

    return D_xx - 2 * D_xy + D_yy


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


class distance():
    def __init__(self, batchsize=None):
        self.N = 96
        anchors = sphere_points(self.N)

        anchors = torch.from_numpy(anchors).float()
        # num, _ = anchors.shape
        M = torch.ones((self.N, self.N))
        for i in range(self.N):
            for j in range(self.N):
                norms = torch.norm((anchors[i] - anchors[j]))
                M[i, j] = norms
        M[M < 0] = 0
        M = M.unsqueeze(0)

        anchors = anchors.unsqueeze(0)
        self.anchors = anchors.repeat(batchsize, 1, 1).cuda()
        self.M = M.repeat(batchsize, 1, 1).cuda()
        # print (self.M)
        # print (1/0)

    def spherical_distance(self, x, y):
        # x = torch.cat((x, self.anchors), 2)
        # y = torch.cat((y, self.anchors), 2)
        y = y.detach()

        D_xx = (x * x).sum(-1).unsqueeze(2)  # (B,N,1)
        D_xy = torch.matmul(x, y.permute(0, 2, 1))  # (B,N,D) @ (B,D,M) = (B,N,M)
        D_yy = (y * y).sum(-1).unsqueeze(1)  # (B,1,M)

        D = (D_xx - 2 * D_xy + D_yy) * 0.1 + self.M.detach()
        # print (D)
        # print (D.shape)
        # print (1/0)

        return D

# def distances(x, y):
#     return sqrt_0( squared_distances(x,y) )
