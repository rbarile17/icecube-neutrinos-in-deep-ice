"""Utility functions and classes."""

import torch
from torch import nn
from torch import Tensor
import numpy as np
import scipy


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


class VonMisesFisher3DLoss(nn.Module):
    """General class for calculating von Mises-Fisher loss.

    Requires implementation for specific dimension `m` in which the target and
    prediction vectors need to be prepared.
    """

    @classmethod
    # pylint: disable=invalid-name
    def log_cmk_exact(
        cls, m: int, kappa: Tensor
    ) -> Tensor:
        """Calculate $log C_{m}(k)$ term in von Mises-Fisher loss exactly."""
        return LogCMK.apply(m, kappa)

    @classmethod
    # pylint: disable=invalid-name
    def log_cmk_approx(
        cls, m: int, kappa: Tensor
    ) -> Tensor:
        """Calculate $log C_{m}(k)$ term in von Mises-Fisher loss approx.

        [https://arxiv.org/abs/1812.04616] Sec. 8.2 with additional minus sign.
        """
        v = m / 2.0 - 0.5
        a = torch.sqrt((v + 1) ** 2 + kappa**2)
        b = v - 1
        return -a + b * torch.log(b + a)

    @classmethod
    # pylint: disable=invalid-name
    def log_cmk(
        cls, m: int, kappa: Tensor, kappa_switch: float = 100.0
    ) -> Tensor:
        """Calculate $log C_{m}(k)$ term in von Mises-Fisher loss.

        Since `log_cmk_exact` is diverges for `kappa` >~ 700 (using float64
        precision), and since `log_cmk_approx` is unaccurate for small `kappa`,
        this method automatically switches between the two at `kappa_switch`,
        ensuring continuity at this point.
        """
        kappa_switch = torch.tensor([kappa_switch]).to(kappa.device)
        mask_exact = kappa < kappa_switch

        # Ensure continuity at `kappa_switch`
        offset = cls.log_cmk_approx(m, kappa_switch) - cls.log_cmk_exact(
            m, kappa_switch
        )
        ret = cls.log_cmk_approx(m, kappa) - offset
        ret[mask_exact] = cls.log_cmk_exact(m, kappa[mask_exact])
        return ret

    # pylint: disable=invalid-name
    def _evaluate(self, prediction: Tensor, target: Tensor) -> Tensor:
        """Calculate von Mises-Fisher loss for a vector in D dimensons.

        This loss utilises the von Mises-Fisher distribution, which is a
        probability distribution on the (D - 1) sphere in D-dimensional space.

        Args:
            prediction: Predicted vector, of shape [batch_size, D].
            target: Target unit vector, of shape [batch_size, D].

        Returns:
            Elementwise von Mises-Fisher loss terms.
        """
        # Check(s)
        assert prediction.dim() == 2
        assert target.dim() == 2
        assert prediction.size() == target.size()

        # Computing loss
        m = target.size()[1]
        k = torch.norm(prediction, dim=1)
        dotprod = torch.sum(prediction * target, dim=1)
        elements = -self.log_cmk(m, k) - dotprod
        return elements

    def forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        """Calculate von Mises-Fisher loss for a direction in the 3D.

        Args:
            prediction: Output of the model. Must have shape [N, 4] where
                columns 0, 1, 2 are predictions of `direction` and last column
                is an estimate of `kappa`.
            target: Target tensor, extracted from graph object.

        Returns:
            Elementwise von Mises-Fisher loss terms. Shape [N,]
        """
        target = target.reshape(-1, 3)
        # Check(s)
        assert prediction.dim() == 2 and prediction.size()[1] == 4
        assert target.dim() == 2
        assert prediction.size()[0] == target.size()[0]

        kappa = prediction[:, 3]
        p = kappa.unsqueeze(1) * prediction[:, [0, 1, 2]]
        return self._evaluate(p, target)
