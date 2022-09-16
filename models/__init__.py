# !/usr/bin/env python3
"""export all available models"""
import segmentation_models_pytorch as smp

# store self-defined models, with both encoder and decoder all together, selected by kwargs "model_name"
__models__ = {}
# store self-defined, smp and timm allowed encoders, decoders, specified by "encoder" and "decoder" kwargs
__smp_encoders__ = smp.encoders.get_encoder_names()
__smp_decoders__ = ["Unet",
                    "UnetPlusPlus",
                    "MAnet",
                    "Linknet",
                    "FPN",
                    "PSPNet",
                    "DeepLabV3",
                    "DeepLabV3Plus",
                    "PAN"]


def get_models(**kwargs):
    """Return the required model object"""
    if "model_name" in kwargs:
        model_name = kwargs["model_name"]
        assert model_name in __models__, f"The model {model_name} does not exist"
        return __models__["model_name"](**kwargs)
    elif "encoder_name" in kwargs and "decoder_name" in kwargs:
        encoder_name = kwargs["encoder_name"]
        decoder_name = kwargs["decoder_name"]
        assert encoder_name in __smp_encoders__, f"The encoder {encoder_name} does not exist"
        assert decoder_name in __smp_decoders__, f"The decoder {decoder_name} does not exist"
        return smp.create_model(arch=decoder_name, encoder_name=encoder_name, in_channels=kwargs["in_channels"],
                                classes=kwargs["in_channels"], **kwargs)

    else:
        raise ModuleNotFoundError("You must specify model_name or encoder decoder names")
