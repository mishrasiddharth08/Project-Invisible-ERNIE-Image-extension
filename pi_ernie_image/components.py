"""ERNIE component factories extracted from Forge's shared loader."""
import os
import torch
import backend.args
from backend import memory_management, utils
from backend.loader import logger
from backend.operations import using_forge_operations
from backend.state_dict import detect_quantization, load_state_dict
from backend.utils import beautiful_print_gguf_state_dict_statics, read_arbitrary_config
from transformers.modeling_utils import no_init_weights
from .transformer import ErnieImageModel


def load_component(guess, component_name, lib_name, cls_name, repo_path, state_dict):
    config_path = os.path.join(repo_path, component_name)

    state_dict_dtype = utils.weight_dtype(state_dict) if isinstance(state_dict, dict) else None

    def _detect_dtype(key: str, storage_dtype: torch.dtype, quant_config: dict = None) -> torch.dtype:
        if quant_config is not None:
            logger.info(f"Using {quant_config.pop('layout')} for {key}")
            return state_dict_dtype
        elif state_dict_dtype in (torch.float8_e4m3fn, torch.float8_e5m2, "gguf"):
            _log = f"{state_dict_dtype}" + (" (pre-quant)" if state_dict_dtype == "gguf" else "")
            logger.info(f"Using Detected {key} Data Type: {_log}")
            if state_dict_dtype == "gguf":
                beautiful_print_gguf_state_dict_statics(state_dict)
            return state_dict_dtype
        else:
            logger.info(f"Using Default {key} Data Type: {storage_dtype}")
            return storage_dtype
    if cls_name == "Mistral3Model":
        if not isinstance(state_dict, dict) or len(state_dict) <= 16:
            raise ValueError("ERNIE Image requires the Ministral 3 3B text encoder. Select ministral-3-3b.safetensors.")

        from .text_encoder import Ministral3_3B

        config = read_arbitrary_config(config_path)

        quant_config = detect_quantization(state_dict)
        storage_dtype = _detect_dtype("Mistral3", memory_management.text_encoder_dtype(), quant_config)

        if storage_dtype == "gguf":
            with no_init_weights():
                with using_forge_operations(device=memory_management.cpu, dtype=memory_management.text_encoder_dtype(), manual_cast_enabled=False, extra_dtype=storage_dtype):
                    model = Ministral3_3B(config)
        else:
            with no_init_weights():
                with using_forge_operations(device=memory_management.cpu, dtype=storage_dtype, manual_cast_enabled=True, sd_dtype=state_dict_dtype, extra_dtype=quant_config):
                    model = Ministral3_3B(config)

        load_state_dict(model, state_dict, log_name=cls_name)
        return model
    if cls_name == "ErnieImageTransformer2DModel":
        if not isinstance(state_dict, dict) or len(state_dict) <= 16:
            raise ValueError("Select an ERNIE Image checkpoint.")
        load_device = memory_management.get_torch_device()
        offload_device = memory_management.unet_offload_device()

        unet_config = guess.unet_config.copy()
        state_dict_parameters = utils.calculate_parameters(state_dict)
        quant_config = "nunchaku" if guess.nunchaku else detect_quantization(state_dict, is_unet=True)

        override_dtype = backend.args.dynamic_args.forge_unet_storage_dtype

        if guess.nunchaku:
            storage_dtype = torch.bfloat16
            logger.info(f"Using Nunchaku Model Data Type: {storage_dtype}")
        elif quant_config is not None:
            storage_dtype = torch.bfloat16
            logger.info(f"Using {quant_config.pop('layout')} for Model")
        elif state_dict_dtype in (torch.float8_e4m3fn, torch.float8_e5m2, "gguf"):
            storage_dtype = state_dict_dtype
            _log = f"{state_dict_dtype}" + (" (pre-quant)" if state_dict_dtype == "gguf" else "")
            logger.info(f"Using Detected Model Data Type: {_log}")
            if state_dict_dtype == "gguf":
                beautiful_print_gguf_state_dict_statics(state_dict)
        else:
            if override_dtype is not None:
                storage_dtype = override_dtype
            else:
                storage_dtype = memory_management.unet_dtype(device=load_device, model_params=state_dict_parameters, supported_dtypes=guess.supported_inference_dtypes, weight_dtype=state_dict_dtype)
            if storage_dtype == state_dict_dtype:
                logger.info(f"Using Default Model Data Type: {storage_dtype}")
            else:
                logger.info(f"Using Override Model Data Type: {storage_dtype}")

        if guess.nunchaku:
            computation_dtype = storage_dtype
        else:
            computation_dtype = memory_management.inference_cast(weight_dtype=storage_dtype, inference_device=load_device, supported_dtypes=guess.supported_inference_dtypes)

        backend.args.dynamic_args.ops = None

        if storage_dtype == "gguf":
            initial_device = memory_management.unet_initial_load_device(parameters=state_dict_parameters, dtype=computation_dtype)
            need_manual_cast = False
            to_args = dict(device=initial_device, dtype=computation_dtype)

            with no_init_weights():
                with using_forge_operations(**to_args, manual_cast_enabled=need_manual_cast, extra_dtype=storage_dtype):
                    model = ErnieImageModel(**unet_config)
        else:
            initial_device = memory_management.unet_initial_load_device(parameters=state_dict_parameters, dtype=storage_dtype)
            need_manual_cast = storage_dtype != computation_dtype
            to_args = dict(device=initial_device, dtype=storage_dtype)

            with no_init_weights():
                with using_forge_operations(**to_args, manual_cast_enabled=need_manual_cast, sd_dtype=state_dict_dtype, extra_dtype=quant_config):
                    model = ErnieImageModel(**unet_config)

        load_state_dict(model, state_dict)

        model.config = unet_config
        model.storage_dtype = storage_dtype
        model.computation_dtype = computation_dtype
        model.load_device = load_device
        model.initial_device = initial_device
        model.offload_device = offload_device

        return model
    raise ValueError(f"Unknown ERNIE component: {cls_name}")
