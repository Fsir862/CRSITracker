import torch
import torch.nn as nn
from torchvision.ops import roi_align


def xywh_to_xyxy(b):
    cx, cy, w, h = b.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)


class NodeEncoder(nn.Module):
    def __init__(self, in_ch, dim, pool_hw=7):
        super().__init__()
        self.pool_hw = pool_hw
        self.proj = nn.Sequential(
            nn.Conv2d(in_ch, dim, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )

    def forward(self, feat_map, boxes_xyxy, stride=4):
        scale_boxes = boxes_xyxy / stride
        rois = roi_align(
            feat_map,
            [scale_boxes],
            output_size=(self.pool_hw, self.pool_hw),
            aligned=True
        )
        emb = self.proj(rois).flatten(1)
        return emb


def build_match_pair(feat_prev, feat_curr, boxes_prev_xywh, p_hat_xy, cfg, encoder=None):
    device = feat_curr.device

    if encoder is None:
        enc = NodeEncoder(cfg.feat_channels, cfg.dim, cfg.pool_hw).to(device)
    else:
        enc = encoder

    b_prev_xyxy = xywh_to_xyxy(boxes_prev_xywh)
    V = enc(feat_prev, b_prev_xyxy, stride=cfg.last_stride)

    cx, cy = p_hat_xy[:, 0], p_hat_xy[:, 1]
    w, h = boxes_prev_xywh[:, 2], boxes_prev_xywh[:, 3]
    cur_xyxy = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)
    U = enc(feat_curr, cur_xyxy, stride=cfg.last_stride)

    r = cfg.alpha * torch.maximum(w, h) + cfg.beta
    r_feat = (r / cfg.last_stride).clamp(min=1.0)

    meta = {
        "centers": p_hat_xy,
        "r_feat": r_feat,
    }
    return U, V, meta