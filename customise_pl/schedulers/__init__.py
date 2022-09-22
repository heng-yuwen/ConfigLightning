from typing import Dict

from .warmup_schedulers import PolyLRScheduler

__all__ = {"PolyLRScheduler": PolyLRScheduler}


def build_scheduler(optimizer, cfg: Dict, num_epochs:int):
    if not isinstance(cfg, Dict):
        return None
    assert "type" in cfg, "Scheduler type is not specified"
    return __all__[cfg.pop("type")](optimizer, num_epochs=num_epochs, **cfg)
