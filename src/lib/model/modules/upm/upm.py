import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import UPMConfig
from .roi import roi_align_fixed, boxes_to_search
from .xcorr import normalized_xcorr
from .heads import RHead


class UncertainDispUPM(nn.Module):
    def __init__(self, cfg: UPMConfig):
        super().__init__()
        self.cfg = cfg
        self.head = RHead(in_ch=1, with_stats=True)

    @staticmethod
    def _xywh_to_xyxy(b):
        cx, cy, w, h = b.unbind(-1)
        x1 = cx - 0.5 * w
        y1 = cy - 0.5 * h
        x2 = cx + 0.5 * w
        y2 = cy + 0.5 * h
        return torch.stack([x1, y1, x2, y2], dim=-1)

    @staticmethod
    def _valid_boxes(boxes_xywh):
        return (boxes_xywh[..., 2] > 1e-3) & (boxes_xywh[..., 3] > 1e-3)

    def _normalize_disp(self, b_prev, b_cur):
        d = b_cur[:, :2] - b_prev[:, :2]
        scale = b_prev[:, 2:].amax(-1, keepdim=True).clamp(min=1e-6)
        return d / scale

    def _denorm_disp(self, b_prev, d_norm):
        scale = b_prev[:, 2:].amax(-1, keepdim=True)
        return d_norm * scale

    def _peak_stats(self, R):
        N = R.size(0)
        if N == 0:
            return R.new_zeros((0, 4))

        B, _, H, W = R.shape
        flat = R.view(B, -1)

        top2_vals, top2_idx = torch.topk(flat, k=2, dim=-1)
        peak = top2_vals[:, 0]
        second = top2_vals[:, 1]
        ratio = peak / (second + 1e-6)

        ys, xs = torch.meshgrid(
            torch.arange(H, device=R.device),
            torch.arange(W, device=R.device),
            indexing='ij'
        )
        coords = torch.stack([xs, ys], dim=-1).view(1, -1, 2).float()

        peak_idx = top2_idx[:, 0:1]
        peak_xy = torch.gather(
            coords.expand(N, -1, -1),
            1,
            peak_idx.unsqueeze(-1).expand(-1, -1, 2)
        ).squeeze(1)

        mask_high = (flat >= (peak * 0.9).unsqueeze(-1)).float()
        num = mask_high.sum(-1).clamp(min=1.0)

        d2 = ((coords.expand(N, -1, -1) - peak_xy.unsqueeze(1)) ** 2).sum(-1)
        width = (d2 * mask_high).sum(-1) / num
        width = torch.sqrt(width + 1e-6)

        return torch.stack([peak, second, ratio, width], dim=-1)

    def _disp_conf_from_stats(self, stats):
        if stats.numel() == 0:
            return stats.new_zeros((0,))

        peak = stats[:, 0]
        second = stats[:, 1]
        width = stats[:, 3]

        sharp = (peak - second).clamp(min=0.0)
        sharp_norm = sharp / (peak.abs() + 1e-6)

        tau = getattr(self.cfg, 'width_tau', 2.0)
        width_factor = torch.exp(-width / tau)

        conf = sharp_norm * width_factor

        target_mean = getattr(self.cfg, 'disp_conf_target_mean', 0.5)
        cur_mean = conf.mean().detach()
        scale = target_mean / (cur_mean + 1e-6)
        conf = (conf * scale).clamp(0.0, 1.0)

        return conf

    def _early_return(self, feat_prev):
        zero = feat_prev.new_tensor(0.0)
        return dict(
            loss=zero,
            p_hat=feat_prev.new_zeros((0, 2)),
            peak_ratio_vec=feat_prev.new_zeros((0,)),
            disp_conf_vec=feat_prev.new_zeros((0,)),
            d_hat=feat_prev.new_zeros((0, 2)),
            d_gt=feat_prev.new_zeros((0, 2)),
        )

    def train_forward(self, feat_prev, feat_curr,
                      boxes_prev_xywh, boxes_cur_xywh,
                      batch_ids=None):
        if boxes_prev_xywh is None or boxes_prev_xywh.numel() == 0:
            return self._early_return(feat_prev)

        device = feat_prev.device
        boxes_prev_xywh = boxes_prev_xywh.to(device)
        boxes_cur_xywh = boxes_cur_xywh.to(device)

        if batch_ids is None:
            batch_ids = torch.zeros(
                boxes_prev_xywh.size(0), device=device, dtype=torch.long)
        else:
            batch_ids = batch_ids.to(device).long()

        valid = self._valid_boxes(boxes_prev_xywh) & self._valid_boxes(boxes_cur_xywh)
        if not valid.any():
            return self._early_return(feat_prev)

        boxes_prev_xywh = boxes_prev_xywh[valid]
        boxes_cur_xywh = boxes_cur_xywh[valid]
        batch_ids = batch_ids[valid]

        T = roi_align_fixed(
            feat_prev,
            self._xywh_to_xyxy(boxes_prev_xywh),
            out_size=self.cfg.template,
            stride=self.cfg.last_stride,
            batch_inds=batch_ids
        )
        S = roi_align_fixed(
            feat_curr,
            boxes_to_search(boxes_prev_xywh, stride=self.cfg.last_stride, S=self.cfg.s_default),
            out_size=self.cfg.search,
            stride=self.cfg.last_stride,
            batch_inds=batch_ids
        )

        if T.numel() == 0 or S.numel() == 0:
            return self._early_return(feat_prev)

        if self.cfg.detach_feat:
            T = T.detach()
            S = S.detach()

        R = normalized_xcorr(S, T, use_ncc=self.cfg.use_ncc)
        stats = self._peak_stats(R)
        peak_ratio_vec = stats[:, 2]
        disp_conf_vec = self._disp_conf_from_stats(stats)

        d_hat = self.head(R, stats=stats)
        d_gt = self._normalize_disp(boxes_prev_xywh, boxes_cur_xywh)
        loss = F.smooth_l1_loss(d_hat, d_gt, reduction='mean')

        d_pixel = self._denorm_disp(boxes_prev_xywh, d_hat)
        p_prev = boxes_prev_xywh[:, :2]
        p_hat = p_prev + d_pixel

        H_feat = feat_curr.size(2)
        W_feat = feat_curr.size(3)
        H_img = H_feat * self.cfg.last_stride
        W_img = W_feat * self.cfg.last_stride
        p_hat_x = p_hat[:, 0].clamp(0, W_img - 1)
        p_hat_y = p_hat[:, 1].clamp(0, H_img - 1)
        p_hat = torch.stack([p_hat_x, p_hat_y], dim=-1)

        return dict(
            loss=loss,
            p_hat=p_hat.detach(),
            peak_ratio_vec=peak_ratio_vec.detach(),
            disp_conf_vec=disp_conf_vec.detach(),
            d_hat=d_hat,
            d_gt=d_gt,
        )

    def infer(self, feat_prev, feat_curr, boxes_prev_xywh, batch_ids=None):
        if boxes_prev_xywh is None or boxes_prev_xywh.numel() == 0:
            device = feat_curr.device
            return {
                'p_hat': torch.zeros(0, 2, device=device),
                'R': torch.zeros(0, 1, self.cfg.search, self.cfg.search, device=device),
                'peak_ratio_vec': torch.zeros(0, device=device),
                'disp_conf_vec': torch.zeros(0, device=device),
            }

        T = roi_align_fixed(
            feat_prev,
            self._xywh_to_xyxy(boxes_prev_xywh),
            self.cfg.template,
            self.cfg.last_stride,
            batch_inds=batch_ids,
        )
        S = roi_align_fixed(
            feat_curr,
            boxes_to_search(boxes_prev_xywh, self.cfg.last_stride, self.cfg.s_default),
            self.cfg.search,
            self.cfg.last_stride,
            batch_inds=batch_ids,
        )

        R = normalized_xcorr(S, T, use_ncc=self.cfg.use_ncc)
        stats = self._peak_stats(R)
        peak_ratio_vec = stats[:, 2].detach()
        disp_conf_vec = self._disp_conf_from_stats(stats)

        d_hat = self.head(R, stats=stats)
        d_pixel = self._denorm_disp(boxes_prev_xywh, d_hat)
        p_prev = boxes_prev_xywh[:, :2]
        p_hat = p_prev + d_pixel

        H_feat = feat_curr.size(2)
        W_feat = feat_curr.size(3)
        H_img = H_feat * self.cfg.last_stride
        W_img = W_feat * self.cfg.last_stride
        p_hat_x = p_hat[:, 0].clamp(0, W_img - 1)
        p_hat_y = p_hat[:, 1].clamp(0, H_img - 1)
        p_hat = torch.stack([p_hat_x, p_hat_y], dim=-1)

        return {
            'p_hat': p_hat,
            'R': R,
            'peak_ratio_vec': peak_ratio_vec,
            'disp_conf_vec': disp_conf_vec,
        }