import torch
import torch.nn as nn

from .layers import MHGAT
from .build import build_match_pair, NodeEncoder
from .inject import gauss_splat, DynStamp, FilmGate


class GraphFuse(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.gat = MHGAT(dim=cfg.dim, dropout=0.1)

        if cfg.inject == "gauss":
            self.injector = "gauss"
        elif cfg.inject == "film":
            self.injector = FilmGate(cfg.dim, cfg.feat_channels)
        else:
            self.injector = DynStamp(cfg.dim, cfg.feat_channels, k=cfg.ksize)

        self.encoder = NodeEncoder(cfg.feat_channels, cfg.dim, cfg.pool_hw)

        self.gate_mlp = nn.Sequential(
            nn.Linear(2 * cfg.dim + 1, cfg.dim),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.dim, 1),
        )
        nn.init.constant_(self.gate_mlp[-1].weight, 0.0)
        nn.init.constant_(self.gate_mlp[-1].bias, 0.0)

        self.proj_z2c = nn.Linear(cfg.dim, cfg.feat_channels, bias=False)

    def _compute_inject_alpha(self, epoch):
        start = float(self.cfg.inject_alpha_start)
        end = float(self.cfg.inject_alpha_end)
        return end if epoch is None else end if start == end else end

    def fuse(self, feat_prev, feat_curr, boxes_prev_xywh, p_hat_xy,
             disp_conf=None, epoch=None):

        if feat_prev is None or feat_curr is None:
            return feat_curr
        if boxes_prev_xywh is None or boxes_prev_xywh.numel() == 0:
            return feat_curr
        if p_hat_xy is None or p_hat_xy.numel() == 0:
            return feat_curr

        if getattr(self.cfg, "detach_feat", False):
            feat_prev = feat_prev.detach()

        U, V, meta = build_match_pair(
            feat_prev, feat_curr,
            boxes_prev_xywh, p_hat_xy,
            self.cfg, encoder=self.encoder
        )
        if U is None or U.size(0) == 0:
            return feat_curr

        U_upd = self.gat(U, V)

        if disp_conf is not None:
            conf_tensor = disp_conf
            if conf_tensor.dim() == 1:
                conf_tensor = conf_tensor.view(-1, 1)
            conf_tensor = conf_tensor.clamp(0.0, 1.0)
        else:
            conf_tensor = U.new_full((U.size(0), 1), 0.5)

        gate_in = torch.cat([U, U_upd, conf_tensor], dim=-1)
        g = torch.sigmoid(self.gate_mlp(gate_in))
        z = (1.0 - g) * U + g * U_upd

        centers = meta["centers"]
        r_feat = meta["r_feat"]
        z_inj = z

        if disp_conf is not None:
            conf_flat = disp_conf.to(U.device).float().view(-1)
            keep = conf_flat >= float(self.cfg.disp_conf_inject_min)
            if keep.sum() == 0:
                return feat_curr
            centers = centers[keep]
            r_feat = r_feat[keep]
            z_inj = z_inj[keep]

        if self.cfg.inject == "gauss":
            z_c = self.proj_z2c(z_inj)
            _, F_strong = gauss_splat(feat_curr, centers, z_c, r_feat, self.cfg)
        elif self.cfg.inject == "film":
            _, F_strong = self.injector(feat_curr, centers, z_inj, self.cfg, r_feat)
        else:
            _, F_strong = self.injector(feat_curr, centers, z_inj, self.cfg, r_feat)

        alpha = self._compute_inject_alpha(epoch)
        if alpha <= 0.0:
            return feat_curr

        delta = F_strong - feat_curr
        s = delta.detach().abs().mean().clamp(min=1e-6)
        s = s * float(self.cfg.inject_delta_clip)
        delta = torch.tanh(delta / s) * s

        F_enh = feat_curr + alpha * delta
        return F_enh