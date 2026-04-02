import torch
from torchvision.ops import roi_align


def boxes_to_search(boxes_xywh, stride, S):
    # boxes: [N,4] (cx,cy,w,h) in pixel
    # return search boxes in pixel (x1,y1,x2,y2)
    cx, cy, w, h = boxes_xywh.unbind(-1)
    L = S * torch.max(w, h)
    x1, y1 = cx - L / 2, cy - L / 2
    x2, y2 = cx + L / 2, cy + L / 2
    return torch.stack([x1, y1, x2, y2], dim=-1)


def roi_align_fixed(feat, boxes_xyxy, out_size, stride,
                    batch_inds=None, aligned=True, sampling_ratio=2):
    """
    feat: [B,C,H,W]
    boxes_xyxy: [N,4]  像素坐标
    batch_inds: [N] or [N,1]，每个 box 属于哪一张图
    """
    B, C, H, W = feat.shape
    device = feat.device

    if boxes_xyxy is None or boxes_xyxy.numel() == 0:
        # 没有 ROI 时，返回空 tensor，避免 roi_align 报错
        return feat.new_zeros((0, C, out_size, out_size))

    if batch_inds is None:
        # 兼容旧逻辑：默认都属于第 0 张图
        batch_inds = torch.zeros((boxes_xyxy.size(0), 1),
                                 device=device, dtype=boxes_xyxy.dtype)
    else:
        # [N] → [N,1]，并保证 dtype / device 一致
        batch_inds = batch_inds.view(-1, 1).to(device=device,
                                               dtype=boxes_xyxy.dtype)

    # rois: [N,5] with (batch_idx, x1, y1, x2, y2)
    rois = torch.cat([batch_inds, boxes_xyxy], dim=-1)

    return roi_align(
        feat,
        rois,
        output_size=(out_size, out_size),
        spatial_scale=1.0 / stride,
        sampling_ratio=sampling_ratio,
        aligned=aligned,
    )  # [N,C,out_size,out_size]
