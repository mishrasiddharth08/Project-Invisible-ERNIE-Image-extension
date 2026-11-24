"""ERNIE model contract extracted from Forge Neo."""
from pathlib import Path
import torch
from huggingface_guess.model_list import BASE, latent

class ErnieImage(BASE):
    huggingface_repo = str(Path(__file__).parent / "assets" / "ERNIE-Image")

    unet_config = {
        "image_model": "ernie",
    }

    sampling_settings = {
        "multiplier": 1000.0,
        "shift": 3.0,
    }

    memory_usage_factor = 10.0

    unet_extra_config = {}
    latent_format = latent.Flux2

    supported_inference_dtypes = [torch.bfloat16, torch.float32]

    vae_key_prefix = ["vae."]
    text_encoder_key_prefix = ["text_encoders."]

    unet_target = "transformer"

    def clip_target(self, state_dict={}):
        return {"ministral3_3b.transformer": "text_encoder"}
