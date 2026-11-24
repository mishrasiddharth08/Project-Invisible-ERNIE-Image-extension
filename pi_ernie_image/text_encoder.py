"""Ministral model definition using Forge shared Llama operations."""
from dataclasses import dataclass, asdict
import torch.nn as nn
from backend.nn.llm.llama import BaseLlama, Llama2_

@dataclass
class Ministral3_3BConfig:
    vocab_size: int = 131072
    hidden_size: int = 3072
    intermediate_size: int = 9216
    num_hidden_layers: int = 26
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    max_position_embeddings: int = 262144
    rms_norm_eps: float = 1e-5
    rope_theta: float = 1000000.0
    transformer_type: str = "llama"
    head_dim = 128
    rms_norm_add = False
    mlp_activation = "silu"
    qkv_bias = False
    rope_dims = None
    q_norm = None
    k_norm = None
    rope_scale = None
    final_norm: bool = True

class Ministral3_3B(BaseLlama, nn.Module):
    def __init__(self, config_dict):
        super().__init__()
        config = Ministral3_3BConfig()

        _config_dict = asdict(config)
        for key, value in _config_dict.items():
            if key in config_dict:
                assert value == config_dict[key]

        self.num_layers = config.num_hidden_layers

        self.model = Llama2_(config)
