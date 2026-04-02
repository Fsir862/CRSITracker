from dataclasses import dataclass


@dataclass
class UPMConfig:
    template: int = 7
    search: int = 11
    s_default: float = 2.0
    use_ncc: bool = True
    detach_feat: bool = True
    last_stride: int = 8

    @classmethod
    def from_opt(cls, opt, last_stride):
        return cls(
            template=opt.upm_template,
            search=opt.upm_search,
            s_default=opt.upm_s_default,
            use_ncc=bool(opt.upm_ncc),
            detach_feat=bool(opt.upm_detach_feat),
            last_stride=last_stride,
        )