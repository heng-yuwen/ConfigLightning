# ConfigLightning
This repo use config to parse the training configs, with pytorch-lightning as the backbone.
Use segmentation experiment:
```python
python train.py fit --config configs/segmentation/xj3_segment_config.yaml
```
use spectral recovery experiment:
```python
python train.py fit --config configs/spectral_recovery/spectral_config.yaml
```
