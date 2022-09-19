from .semantic_segmentation import SemanticSegmentor

__all__ = {"SemanticSegmentor": SemanticSegmentor}


def get_experiment(experiment_name):
    assert experiment_name in __all__, f"The experiment {experiment_name} does not exist!"
    return __all__[experiment_name]
