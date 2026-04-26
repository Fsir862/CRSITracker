import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def make_gaussian_kernel(ksize=7, sigma=1.5, dtype=torch.float32, device='cpu'):
    assert ksize % 2 == 1, "ksize must be odd"
    ax = torch.arange(ksize, dtype=dtype, device=device) - (ksize - 1) / 2.0
    xx, yy = torch.meshgrid(ax, ax, indexing='xy')
    kernel = torch.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
    kernel = kernel / kernel.sum()
    return kernel.view(1, 1, ksize, ksize)


class MaskedChannelAttention(nn.Module):
    """
    输入:
        Ft      : B x C x H x W
        H_hat   : B x C_h x H x W or B x 1 x H x W

    输出:
        Ft_enh  : B x C x H x W
        Mc_gauss: B x C x 1 x 1
    """
    def __init__(
        self,
        channels,
        reduction=16,
        ksize=7,
        sigma=1.5,
        detach_mask=False,
        mask_adaptive=False,
        sigma_scale=0.6,
        sigma_min=1.0,
        sigma_max=20.0,
        soft_radius=2.0,
        comp_temp=2.5,
        comp_bias=-0.40,
        gate_pow=2.0,
        mca_base=1.5,
        mca_scale=0.5,
    ):
        super().__init__()
        self.channels = int(channels)
        self.reduction = int(reduction)
        self.ksize = int(ksize)
        self.base_sigma = float(sigma)
        self.detach_mask = bool(detach_mask)

        self.mask_adaptive = bool(mask_adaptive)
        self.sigma_scale = float(sigma_scale)
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.soft_radius = float(soft_radius)
        self.gate_pow = float(gate_pow)
        self.comp_temp = nn.Parameter(torch.tensor(float(comp_temp), dtype=torch.float32))
        self.comp_bias = nn.Parameter(torch.tensor(float(comp_bias), dtype=torch.float32))
        self.mca_base = nn.Parameter(torch.tensor(float(mca_base), dtype=torch.float32))
        self.mca_scale = nn.Parameter(torch.tensor(float(mca_scale), dtype=torch.float32))
        mid_c = max(1, self.channels // self.reduction)
        self.mlp = nn.Sequential(
            nn.Linear(self.channels, mid_c, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(mid_c, self.channels, bias=True),
            nn.Sigmoid()
        )

        self.register_buffer('_kernel', torch.zeros(1, 1, 1, 1))

    def _get_kernel(self, P_norm):
        device = P_norm.device
        dtype = P_norm.dtype

        if self.mask_adaptive:
            B, _, H, W = P_norm.shape
            with torch.no_grad():
                area_ratio = P_norm.view(B, -1).mean(dim=1)
                r_est = torch.sqrt(area_ratio * (H * W) / math.pi)
                sigma = self.sigma_scale * r_est + self.soft_radius
                sigma = torch.clamp(sigma, min=self.sigma_min, max=self.sigma_max)
                sigma_eff = float(sigma.mean().item())

            ksize = int(max(3, min(self.ksize, 2 * int(3 * sigma_eff) + 1)))
            if self._kernel.shape[-1] != ksize:
                self._kernel = make_gaussian_kernel(
                    ksize=ksize,
                    sigma=sigma_eff,
                    dtype=dtype,
                    device=device
                )
        else:
            ksize = self.ksize
            if self._kernel.shape[-1] != ksize:
                self._kernel = make_gaussian_kernel(
                    ksize=ksize,
                    sigma=self.base_sigma,
                    dtype=dtype,
                    device=device
                )

        return self._kernel

    def forward(self, Ft, H_hat):
        if H_hat is None:
            return Ft, None

        B, C, H, W = Ft.shape
        device, dtype = Ft.device, Ft.dtype

        if H_hat.dim() == 4 and H_hat.size(1) > 1:
            P = H_hat.sum(dim=1, keepdim=True)
        else:
            P = H_hat

        if self.detach_mask:
            P = P.detach()

        P = P.to(device=device, dtype=dtype).clamp(min=0.0)

        maxv = P.view(B, -1).max(dim=1)[0].view(B, 1, 1, 1) + 1e-6
        P_norm = P / maxv

        kernel = self._get_kernel(P_norm)
        k = kernel.shape[-1]
        G = F.conv2d(P_norm, kernel, padding=k // 2)
        maxg = G.view(B, -1).max(dim=1)[0].view(B, 1, 1, 1) + 1e-6
        G = (G / maxg).clamp(0.0, 1.0)

        Gp = G.pow(self.gate_pow)
        invGp = (1.0 - G).clamp(0.0, 1.0)

        Gp_sum = Gp.view(B, 1, -1).sum(dim=2) + 1e-6
        invGp_sum = invGp.view(B, 1, -1).sum(dim=2) + 1e-6

        F_in = (Ft * Gp).view(B, C, -1).sum(dim=2) / Gp_sum.view(B, 1)
        F_out = (Ft * invGp).view(B, C, -1).sum(dim=2) / invGp_sum.view(B, 1)

        d = F_in - F_out
        d = d / (d.abs().mean(dim=1, keepdim=True) + 1e-6)
        d = F.layer_norm(d, [C])

        s0 = self.mlp(d).clamp(1e-4, 1.0 - 1e-4)
        z = torch.log(s0) - torch.log(1.0 - s0)
        z = z - z.mean(dim=1, keepdim=True)
        s = torch.sigmoid(self.comp_temp * z + self.comp_bias).view(B, C, 1, 1)

        Mc_gauss = (self.mca_base + self.mca_scale * s)
        Mc_gauss = Mc_gauss.clamp(min=1.0, max=3.0)
        Ft_enh = Ft * Mc_gauss

        return Ft_enh, Mc_gauss
