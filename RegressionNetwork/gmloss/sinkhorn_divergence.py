# coding=utf-8
"""Implements the (unbiased) Sinkhorn divergence between abstract measures.
"""
import numpy as np
import torch
from .utils import scal


def max_diameter(x, y):
    mins = torch.stack((x.min(dim=0)[0], y.min(dim=0)[0])).min(dim=0)[0]
    maxs = torch.stack((x.max(dim=0)[0], y.max(dim=0)[0])).max(dim=0)[0]
    # print (x.shape)
    # print (x.min(dim=0))  #[-0.0389, -0.9750, -0.9901, -0.9844]
    # print (mins.shape) # 4
    diameter = (maxs - mins).norm().item()
    # print (diameter) $ 3.45
    # print (1 / 0)
    return diameter


def epsilon_schedule(p, diameter, blur, scaling):
    ε_s = [diameter ** p] \
          + [np.exp(e) for e in np.arange(p * np.log(diameter), p * np.log(blur), p * np.log(scaling))] \
          + [blur ** p]
    return ε_s


def scaling_parameters(x, y, p, blur, reach, diameter, scaling):
    if diameter is None:
        D = x.shape[-1]
        diameter = max_diameter(x.view(-1, D), y.view(-1, D))

    ε = blur ** p
    ε_s = epsilon_schedule(p, diameter, blur, scaling)
    ρ = None if reach is None else reach ** p
    return diameter, ε, ε_s, ρ


# ==============================================================================
#                              Sinkhorn loop
# ==============================================================================

def dampening(ε, ρ):
    return 1 if ρ is None else 1 / (1 + ε / ρ)


def log_weights(α):
    α_log = α.log()
    α_log[α <= 0] = -100000
    return α_log


class UnbalancedWeight(torch.nn.Module):
    def __init__(self, ε, ρ):
        super(UnbalancedWeight, self).__init__()
        self.ε, self.ρ = ε, ρ

    def forward(self, x):
        return (self.ρ + self.ε / 2) * x

    def backward(self, g):
        return (self.ρ + self.ε) * g


def sinkhorn_cost(ε, ρ, α, β, a_x, b_y, a_y, b_x):
    # Just return the dual potentials
    # Actually compute the Sinkhorn divergence
    # UNBIASED Sinkhorn divergence, S_ε(α,β) = OT_ε(α,β) - .5*OT_ε(α,α) - .5*OT_ε(β,β)  # if ρ is None:
    return scal(α, b_x - a_x) + scal(β, a_y - b_y)


def sinkhorn_loop(softmin, α_log, β_log, C_xx, C_yy, C_xy, C_yx, ε_s, ρ):

    torch.autograd.set_grad_enabled(False)

    k = 0  # Scale index; we start at the coarsest resolution available
    ε = ε_s[k]
    λ = dampening(ε, ρ)

    # Load the measures and cost matrices at the current scale:
    # Start with a decent initialization for the dual vectors:
    a_x = λ * softmin(ε, C_xx, α_log)  # OT(α,α)
    b_y = λ * softmin(ε, C_yy, β_log)  # OT(β,β)
    a_y = λ * softmin(ε, C_yx, α_log)  # OT(α,β) wrt. a
    b_x = λ * softmin(ε, C_xy, β_log)  # OT(α,β) wrt. b

    for i, ε in enumerate(ε_s):  # ε-scaling descent
        λ = dampening(ε, ρ)  # ε has changed, so we should update λ too!
        # "Coordinate ascent" on the dual problems:
        at_x = λ * softmin(ε, C_xx, α_log + a_x / ε)  # OT(α,α)
        bt_y = λ * softmin(ε, C_yy, β_log + b_y / ε)  # OT(β,β)
        at_y = λ * softmin(ε, C_yx, α_log + b_x / ε)  # OT(α,β) wrt. a
        bt_x = λ * softmin(ε, C_xy, β_log + a_y / ε)  # OT(α,β) wrt. b

        # Symmetrized updates:
        a_x, b_y = .5 * (a_x + at_x), .5 * (b_y + bt_y)  # OT(α,α), OT(β,β)
        a_y, b_x = .5 * (a_y + at_y), .5 * (b_x + bt_x)  # OT(α,β) wrt. a, b

    torch.autograd.set_grad_enabled(True)

    # Last extrapolation, to get the correct gradients:
    a_x = λ * softmin(ε, C_xx, (α_log + a_x / ε).detach())
    b_y = λ * softmin(ε, C_yy, (β_log + b_y / ε).detach())

    # The cross-updates should be done in parallel!
    a_y, b_x = λ * softmin(ε, C_yx, (α_log + b_x / ε).detach()), \
               λ * softmin(ε, C_xy, (β_log + a_y / ε).detach())

    return a_x, b_y, a_y, b_x
