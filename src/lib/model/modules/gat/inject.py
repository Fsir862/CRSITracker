import torch
import torch.nn as nn
import math
def _xy_to_feat(xy_pix, stride):
    return xy_pix / stride

def _scatter_add_patch(F, patch, cx, cy):
    """
    F: [1, C, H, W]
    patch: [C, kh, kw]
    (cx, cy): float，特征图坐标（不是像素坐标）
    双线性写回：将 patch 的每个格点按四邻插值分配到 F 上。
    目前假设 batch=1。
    """
    B, C, H, W = F.shape
    assert B == 1, "当前实现假定 batch=1"
    _, kh, kw = patch.shape

    device = F.device
    dtype = F.dtype

    # 1) 生成每个 patch 元素在特征图上的浮点坐标
    # 这里让 (cx,cy) 对齐到 patch 中心
    # 注意用 (kw-1)/2, (kh-1)/2 会更对称一些
    x0f = cx - (kw - 1) * 0.5
    y0f = cy - (kh - 1) * 0.5

    gy = torch.arange(kh, device=device, dtype=dtype)
    gx = torch.arange(kw, device=device, dtype=dtype)
    yf, xf = torch.meshgrid(gy, gx)  # [kh, kw], [kh, kw]
    xf = x0f + xf
    yf = y0f + yf

    # 2) 四邻整数坐标
    x0 = torch.floor(xf)
    y0 = torch.floor(yf)
    x1 = x0 + 1.0
    y1 = y0 + 1.0

    # 3) 双线性权重
    wx1 = (xf - x0).clamp(0, 1)
    wy1 = (yf - y0).clamp(0, 1)
    wx0 = 1.0 - wx1
    wy0 = 1.0 - wy1

    w00 = (wx0 * wy0).view(-1)  # [kh*kw]
    w10 = (wx1 * wy0).view(-1)
    w01 = (wx0 * wy1).view(-1)
    w11 = (wx1 * wy1).view(-1)

    x0 = x0.view(-1).long()
    y0 = y0.view(-1).long()
    x1 = x1.view(-1).long()
    y1 = y1.view(-1).long()

    patch_flat = patch.view(C, -1)  # [C, kh*kw]
    F2 = F.view(C, -1)              # [C, H*W]

    def add_one(xi, yi, wi):
        # xi, yi: [K] long; wi: [K] float
        # 4) 边界裁剪
        in_x = (xi >= 0) & (xi < W)
        in_y = (yi >= 0) & (yi < H)
        valid = in_x & in_y
        if not valid.any():
            return

        xi = xi[valid]
        yi = yi[valid]
        wi = wi[valid]

        flat_idx = (yi * W + xi)  # [K_valid]
        # 对应 patch 中的元素（和 wi 一一对应）
        patch_sel = patch_flat[:, valid]        # [C, K_valid]
        pv = patch_sel * wi.view(1, -1)        # [C, K_valid]

        idx = flat_idx.unsqueeze(0).expand(C, -1)  # [C, K_valid]
        F2.scatter_add_(1, idx, pv)

    # 5) 四个邻居分别写回
    add_one(x0, y0, w00)
    add_one(x1, y0, w10)
    add_one(x0, y1, w01)
    add_one(x1, y1, w11)




# ---------- A) Gaussian splat (更保守的小目标版本) ----------
def gauss_splat(F_t, centers_pix, z, r_feat, cfg):
    device = F_t.device
    B, C, H, W = F_t.shape
    assert B == 1

    F_delta = torch.zeros_like(F_t)
    denom   = torch.zeros(1, 1, H, W, device=device)

    if r_feat is not None and r_feat.numel() > 0:
        r_vec = r_feat.float().view(-1)
        r_norm = r_vec / (r_vec.max() + 1e-6)

        size_min   = float(getattr(cfg, "size_factor_min", 0.4))  
        size_max   = float(getattr(cfg, "size_factor_max", 1.0))  
        size_gamma = float(getattr(cfg, "size_factor_gamma", 1.0))

        size_factor = (1.0 - r_norm).pow(size_gamma)
        size_factor = size_factor * (size_max - size_min) + size_min
    else:
        size_factor = None

    sigma_scale = float(getattr(cfg, "gauss_sigma_scale", 1.0))   
    sigma_min   = float(getattr(cfg, "gauss_sigma_min",   0.5))   
    sigma_max   = float(getattr(cfg, "gauss_sigma_max",   3.0))   
    rad_min     = int(getattr(cfg, "gauss_rad_min",       1))     
    rad_max     = int(getattr(cfg, "gauss_rad_max",       5))     
    k_sigma     = float(getattr(cfg, "gauss_k_sigma",     3.0))   

    for i in range(centers_pix.size(0)):
        c = _xy_to_feat(centers_pix[i], cfg.last_stride)
        base_sigma = float(r_feat[i].item())
        sigma = base_sigma * sigma_scale
        sigma = max(sigma_min, min(sigma_max, sigma))
        rad   = int(max(rad_min, min(rad_max, math.ceil(k_sigma * sigma))))
        if rad <= 0:
            continue  
        grid_y = torch.arange(-rad, rad+1, device=device).view(-1, 1).float()
        grid_x = torch.arange(-rad, rad+1, device=device).view(1, -1).float()
        ker = torch.exp(-(grid_x**2 + grid_y**2) / (2.0 * sigma**2)) 
        ker = ker / (ker.sum() + 1e-6)

        patch = z[i].view(C, 1, 1) * ker.unsqueeze(0) 
        if size_factor is not None:
            sf = float(size_factor[i].item())
            patch = patch * sf

        _scatter_add_patch(F_delta, patch, c[0], c[1])
        _scatter_add_patch(denom, ker.unsqueeze(0), c[0], c[1])


    if cfg.do_norm_overlap:
        F_delta = F_delta / (denom.clamp(min=1e-6))

    F_enh = F_t + F_delta
    return F_delta, F_enh
