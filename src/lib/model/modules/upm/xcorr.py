import torch
import torch.nn.functional as F


def depthwise_xcorr_batch(S, T):
    N, C, Hs, Ws = S.shape
    _, _, hT, wT = T.shape
    S_ = S.reshape(1, N * C, Hs, Ws)
    T_ = T.reshape(N * C, 1, hT, wT)
    out = F.conv2d(S_, T_, groups=N * C)
    out = out.view(N, C, out.size(-2), out.size(-1))
    return out.sum(1, keepdim=True)


def local_patch_energy(S, ksize):
    ones = torch.ones(
        (S.size(1), 1, ksize, ksize),
        device=S.device,
        dtype=S.dtype
    )
    S2 = S * S
    S2_ = S2.reshape(1, S.size(0) * S.size(1), S.size(2), S.size(3))
    ones_ = ones.repeat(S.size(0), 1, 1, 1).reshape(S.size(0) * S.size(1), 1, ksize, ksize)
    en = F.conv2d(S2_, ones_, groups=S.size(0) * S.size(1))
    en = en.view(S.size(0), S.size(1), en.size(-2), en.size(-1)).sum(1, keepdim=True)
    return torch.sqrt(torch.clamp(en, min=1e-6))


def normalized_xcorr(S, T, use_ncc=True):
    N = T.size(0)
    T_flat = T.view(N, -1)
    T_norm = T_flat.norm(p=2, dim=1, keepdim=True).clamp(min=1e-6)
    T_hat = (T_flat / T_norm).view_as(T)

    num = depthwise_xcorr_batch(S, T_hat)

    if not use_ncc:
        return torch.tanh(num)

    den = T_norm.view(N, 1, 1, 1) * local_patch_energy(S, ksize=T.size(-1))
    R = num / (den + 1e-6)
    return torch.tanh(R)