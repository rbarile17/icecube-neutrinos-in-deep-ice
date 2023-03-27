"""Utility functions and classes."""

import torch
import numpy as np
import scipy


def eps_like(tensor: torch.Tensor) -> torch.Tensor:
    """Return `eps` matching `tensor`'s dtype."""
    return torch.finfo(tensor.dtype).eps


def angle_to_xyz(angles_b):
    """Converts azimuth and zenith angles to cartesian coordinates."""
    azimuth, zenith = angles_b.t()

    return torch.stack([
        torch.cos(azimuth) * torch.sin(zenith),
        torch.sin(azimuth) * torch.sin(zenith),
        torch.cos(zenith)], dim=1)


def xyz_to_angle(xyz_b):
    """Converts cartesian coordinates to azimuth and zenith angles."""
    dom_x, dom_y, dom_z = xyz_b.t()
    azimuth = torch.arccos(dom_x / torch.sqrt(dom_x**2 + dom_y**2)) * torch.sign(dom_y)
    zenith = torch.arccos(dom_z / torch.sqrt(dom_x**2 + dom_y**2 + dom_z**2))
    return torch.stack([azimuth, zenith], dim=1)


def angular_error(xyz_pred_b, xyz_true_b):
    """Calculates angular error between two sets of vectors."""
    return torch.arccos(torch.sum(xyz_pred_b * xyz_true_b, dim=1))


class LogCMK(torch.autograd.Function):
    """MIT License.

    Copyright (c) 2019 Max Ryabinin

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
    _____________________

    From [https://github.com/mryab/vmf_loss/blob/master/losses.py]
    Modified to use modified Bessel function instead of exponentially scaled ditto
    (i.e. `.ive` -> `.iv`) as indiciated in [1812.04616] in spite of suggestion in
    Sec. 8.2 of this paper. The change has been validated through comparison with
    exact calculations for `m=2` and `m=3` and found to yield the correct results.
    """

    @staticmethod
    # pylint: disable=invalid-name,arguments-differ
    def forward(ctx, m, kappa):
        """Forward pass."""
        dtype = kappa.dtype
        ctx.save_for_backward(kappa)
        ctx.m = m
        ctx.dtype = dtype
        kappa = kappa.double()
        iv = torch.from_numpy(scipy.special.iv(m / 2.0 - 1, kappa.cpu().numpy())).to(kappa.device)
        return (
            (m / 2.0 - 1) * torch.log(kappa)
            - torch.log(iv)
            - (m / 2) * np.log(2 * np.pi)
        ).type(dtype)

    @staticmethod
    # pylint: disable=invalid-name,arguments-differ
    def backward(ctx, grad_output):
        """Backward pass."""
        kappa = ctx.saved_tensors[0]
        m = ctx.m
        dtype = ctx.dtype
        kappa = kappa.double().cpu().numpy()
        grads = -(
            (scipy.special.iv(m / 2.0, kappa)) / (scipy.special.iv(m / 2.0 - 1, kappa))
        )
        return (
            None,
            grad_output * torch.from_numpy(grads).to(grad_output.device).type(dtype),
        )

    @staticmethod
    # pylint: disable=arguments-differ
    def jvp(ctx):
        """See base class."""
        raise NotImplementedError
