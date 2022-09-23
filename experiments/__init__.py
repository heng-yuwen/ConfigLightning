from .semantic_segmentation import SemanticSegmentor
from .spectral_recovery import SpectralRecovery

__all__ = {"SemanticSegmentor": SemanticSegmentor,
           "SpectralRecovery": SpectralRecovery}


def get_experiment(experiment_name):
    assert experiment_name in __all__, f"The experiment {experiment_name} does not exist!"
    return __all__[experiment_name]
