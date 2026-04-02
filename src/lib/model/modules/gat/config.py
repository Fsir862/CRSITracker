from dataclasses import dataclass


@dataclass
class GATConfig:
    enable: bool = False
    last_stride: int = 4
    feat_channels: int = 64

    dim: int = 128
    alpha: float = 1.5
    beta: float = 1.0
    pool_hw: int = 7

    inject: str = "dconv"
    ksize: int = 3
    do_norm_overlap: bool = True
    detach_feat: bool = True

    inject_alpha_start: float = 0.1
    inject_alpha_end: float = 0.1
    inject_delta_clip: float = 2.0

    disp_conf_inject_min: float = 0.25

    gauss_sigma_scale: float = 1.0
    gauss_sigma_min: float = 0.5
    gauss_sigma_max: float = 2.5
    gauss_rad_min: int = 1
    gauss_rad_max: int = 5
    gauss_k_sigma: float = 3.0

    size_factor_min: float = 0.4
    size_factor_max: float = 1.0
    size_factor_gamma: float = 1.0

    @classmethod
    def from_opt(cls, opt, last_stride, feat_channels):
        return cls(
            enable=bool(getattr(opt, "gat", False)),
            last_stride=last_stride,
            feat_channels=feat_channels,

            dim=int(getattr(opt, "gat_dim", 128)),
            alpha=float(getattr(opt, "gat_alpha", 0.2)),
            beta=float(getattr(opt, "gat_beta", 1.0)),
            pool_hw=int(getattr(opt, "gat_pool", 7)),

            inject=str(getattr(opt, "gat_inject", "dconv")),
            ksize=int(getattr(opt, "gat_k", 3)),
            do_norm_overlap=bool(getattr(opt, "gat_norm", 1)),
            detach_feat=bool(getattr(opt, "gat_detach_feat", 1)),

            inject_alpha_start=float(getattr(opt, "gat_inject_alpha_start", 0.1)),
            inject_alpha_end=float(getattr(opt, "gat_inject_alpha_end", 0.1)),
            inject_delta_clip=float(getattr(opt, "gat_inject_delta_clip", 2.0)),

            disp_conf_inject_min=float(getattr(opt, "disp_conf_inject_min", 0.50)),

            gauss_sigma_scale=float(getattr(opt, "gauss_sigma_scale", 1.0)),
            gauss_sigma_min=float(getattr(opt, "gauss_sigma_min", 0.5)),
            gauss_sigma_max=float(getattr(opt, "gauss_sigma_max", 2.5)),
            gauss_rad_min=int(getattr(opt, "gauss_rad_min", 1)),
            gauss_rad_max=int(getattr(opt, "gauss_rad_max", 5)),
            gauss_k_sigma=float(getattr(opt, "gauss_k_sigma", 3.0)),

            size_factor_min=float(getattr(opt, "gat_size_factor_min", 0.01)),
            size_factor_max=float(getattr(opt, "gat_size_factor_max", 1.0)),
            size_factor_gamma=float(getattr(opt, "gat_size_gamma", 1.5)),
        )