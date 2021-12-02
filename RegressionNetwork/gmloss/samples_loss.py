# coding=utf-8
import torch
from torch.nn import Module
from functools import partial
import warnings

import numpy as np
from .utils import squared_distances, distance
from .sinkhorn_divergence import scaling_parameters, log_weights, sinkhorn_cost, sinkhorn_loop


class SamplesLoss(Module):
    """Creates a criterion that computes distances between sampled measures on a vector space.
    Warning:
        If **loss** is ``"sinkhorn"`` and **reach** is **None** (balanced Optimal Transport),
        the resulting routine will expect measures whose total masses are equal with each other.
    Parameters:
              - ``"sinkhorn"``: (Un-biased) Sinkhorn divergence, which interpolates
                between Wasserstein (blur=0) and kernel (blur= :math:`+\infty` ) distances.
    """

    def __init__(self, loss="sinkhorn", p=2, blur=.05, reach=None, diameter=None, scaling=.5, batchsize=None):

        super(SamplesLoss, self).__init__()
        self.loss = loss
        self.p = p
        self.blur = blur
        self.reach = reach
        self.diameter = diameter
        self.scaling = scaling

        # self.shperical_distance = self.distance.spherical_distance()

    def forward(self, x, y, geometry):
        """Computes the loss between sampled measures.
        Documentation and examples: Soon!
        Until then, please check the tutorials :-)"""

        α, x, β, y = self.process_args(x, y)
        # B, N, M, D = self.check_shapes(l_x, α, x, l_y, β, y)
        values = self.sinkhorn_tensorized(α, x, β, y, p=self.p,
                                          blur=self.blur, reach=self.reach, diameter=self.diameter,
                                          scaling=self.scaling, geometry=geometry)

        return values  # The user expects a "batch vector" of distances

    def process_args(self, x, y):
        α = self.generate_weights(x)
        β = self.generate_weights(y)
        return α, x, β, y


    @staticmethod
    def generate_weights(x):
        if x.dim() == 2:  #
            N = x.shape[0]
            return torch.ones(N).type_as(x) / N
        elif x.dim() == 3:
            B, N, _ = x.shape
            return torch.ones(B, N).type_as(x) / N
        else:
            raise ValueError("Input samples 'x' and 'y' should be encoded as (N,D) or (B,N,D) (batch) tensors.")

    @staticmethod
    def softmin_tensorized(ε, C, wlog):
        B = C.shape[0]
        return - ε * (wlog.view(B, 1, -1) - C / ε).logsumexp(2).view(B, -1)

    def sinkhorn_tensorized(self, α, x, β, y, p=2, blur=.05, reach=None, diameter=None, scaling=.5, geometry=None):
        B, M, _ = y.shape

        self.distance = distance(batchsize=B, geometry=geometry)

        cost = (lambda m, n: self.distance.geometric_distance(m, n) / 2)
        # cost = (lambda m, n: squared_distances(m, n) / 2)

        C_xx, C_yy = (cost(x, x.detach()), cost(y, y.detach()))  # (B,N,N), (B,M,M)
        C_xy, C_yx = (cost(x, y.detach()), cost(y, x.detach()))  # (B,N,M), (B,M,N)

        diameter, ε, ε_s, ρ = scaling_parameters(x, y, p, blur, reach, diameter, scaling)
        a_x, b_y, a_y, b_x = sinkhorn_loop(self.softmin_tensorized, log_weights(α), log_weights(β),
                                           C_xx, C_yy, C_xy, C_yx, ε_s, ρ)

        return sinkhorn_cost(ε, ρ, α, β, a_x, b_y, a_y, b_x)
