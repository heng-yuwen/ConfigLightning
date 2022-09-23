from typing import Dict
from torch.optim.lr_scheduler import CosineAnnealingLR
from .warmup_schedulers import PolyLRScheduler

__all__ = {"PolyLRScheduler": PolyLRScheduler,
           "CosineAnnealingLR": CosineAnnealingLR}


def build_scheduler(optimizer, cfg: Dict, num_epochs: int):
    if not isinstance(cfg, Dict):
        return None
    assert "type" in cfg, "Scheduler type is not specified"
    if "by_iteration" in cfg and cfg.pop("by_iteration"):
        total_iteration = num_epochs * 1000
        return __all__[cfg.pop("type")](optimizer, total_iteration, **cfg)
    else:
        return __all__[cfg.pop("type")](optimizer, num_epochs=num_epochs, **cfg)
