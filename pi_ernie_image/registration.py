"""Register ERNIE through Forge's shared detection and loading lifecycle."""
from functools import wraps
from inspect import signature
from logging import getLogger
import sys

logger = getLogger(__name__)
_installed = []


def unregister():
    """Undo hooks still owned by this extension."""
    from backend import loader
    from huggingface_guess import model_list
    from .engine import ErnieImage
    from .model_config import ErnieImage as ErnieConfig

    for module, name, original, replacement in reversed(_installed):
        if getattr(module, name) is replacement:
            setattr(module, name, original)
    _installed.clear()
    loader.possible_models = tuple(cls for cls in loader.possible_models if cls is not ErnieImage)
    model_list.models[:] = [cls for cls in model_list.models if cls is not ErnieConfig]


def register() -> bool:
    from backend import loader
    from backend.patcher import lora as forge_lora
    from huggingface_guess import detection, model_list
    from modules_forge.packages.comfy import lora
    from .components import load_component
    from .engine import ErnieImage
    from .model_config import ErnieImage as ErnieConfig

    if _installed:
        return True
    if any(cls.__name__ == "ErnieImage" for cls in loader.possible_models):
        logger.info("Project Invisible ERNIE Image: Forge's native ERNIE engine is active.")
        return True

    detect = detection.detect_unet_config
    component = loader.load_huggingface_component
    replace = loader.replace_state_dict
    adapter = lora.model_lora_keys_unet
    for function, parameters in (
        (detect, ("state_dict", "key_prefix")),
        (component, ("guess", "component_name", "lib_name", "cls_name", "repo_path", "state_dict")),
        (adapter, ("model",)),
    ):
        signature(function).bind_partial(**dict.fromkeys(parameters))

    @wraps(detect)
    def detect_ernie(state_dict, key_prefix, *args, **kwargs):
        if f"{key_prefix}layers.0.mlp.linear_fc2.weight" in state_dict:
            return {"image_model": "ernie"}
        return detect(state_dict, key_prefix, *args, **kwargs)

    @wraps(component)
    def load_ernie(guess, component_name, lib_name, cls_name, repo_path, state_dict):
        if isinstance(guess, ErnieConfig):
            if component_name == "vae":
                weight = (state_dict or {}).get("decoder.conv_in.weight")
                if weight is None or weight.shape[1] != 32:
                    raise ValueError("ERNIE Image requires the FLUX.2 VAE. Select flux2-vae.safetensors.")
            if cls_name in ("Mistral3Model", "ErnieImageTransformer2DModel"):
                return load_component(guess, component_name, lib_name, cls_name, repo_path, state_dict)
        return component(guess, component_name, lib_name, cls_name, repo_path, state_dict)

    @wraps(replace)
    def replace_ernie(sd, asd, *args, **kwargs):
        guess = kwargs["guess"] if "guess" in kwargs else args[-1 if "path" in kwargs else -2]
        if isinstance(guess, ErnieConfig) and "model.layers.0.post_attention_layernorm.weight" in asd:
            weight = asd["model.layers.0.post_attention_layernorm.weight"]
            if weight.shape[0] != 3072 or "model.layers.0.self_attn.q_norm.weight" in asd:
                raise ValueError("ERNIE Image requires the Ministral 3 3B text encoder.")
            prefix = f"{guess.text_encoder_key_prefix[0]}ministral3_3b.transformer."
            for key in list(sd):
                if key.startswith(prefix):
                    del sd[key]
            sd.update({prefix + key: value for key, value in asd.items() if key.startswith("model")})
            return sd
        return replace(sd, asd, *args, **kwargs)

    @wraps(adapter)
    def ernie_adapter(model, key_map=None):
        result = adapter(model, {} if key_map is None else key_map)
        if isinstance(model.config, ErnieConfig):
            for key in model.state_dict():
                if key.startswith("diffusion_model.") and key.endswith(".weight"):
                    result["transformer." + key[len("diffusion_model."):-len(".weight")]] = key
        return result

    hooks = [
        (detection, "detect_unet_config", detect, detect_ernie),
        (loader, "load_huggingface_component", component, load_ernie),
        (loader, "replace_state_dict", replace, replace_ernie),
        (lora, "model_lora_keys_unet", adapter, ernie_adapter),
    ]
    if forge_lora.model_lora_keys_unet is adapter:
        hooks.append((forge_lora, "model_lora_keys_unet", adapter, ernie_adapter))
    networks = sys.modules.get("networks")
    if networks is not None and getattr(networks, "model_lora_keys_unet", None) is adapter:
        hooks.append((networks, "model_lora_keys_unet", adapter, ernie_adapter))
    model_list.models.append(ErnieConfig)
    loader.possible_models = (*loader.possible_models, ErnieImage)
    for module, name, original, replacement in hooks:
        setattr(module, name, replacement)
    _installed.extend(hooks)
    logger.info("Project Invisible ERNIE Image: independent backup engine registered.")
    return True
