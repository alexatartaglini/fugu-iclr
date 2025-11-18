"""
Each modeling file in this library is a mapping between
abstract naming of intervention anchor points and actual
model module defined in the huggingface library.

We also want to let the intervention library know how to
config the dimensions of intervention based on model config
defined in the huggingface library.
"""


import torch
from ..constants import *


internvl_type_to_module_mapping = {
    "language.block_input": ("language_model.model.layers[%s]", CONST_INPUT_HOOK),
    "language.block_output": ("language_model.model.layers[%s]", CONST_OUTPUT_HOOK),
    "language.mlp_activation": ("language_model.model.layers[%s].mlp.act_fn", CONST_OUTPUT_HOOK),
    "language.mlp_output": ("language_model.model.layers[%s].mlp", CONST_OUTPUT_HOOK),
    "language.mlp_input": ("language_model.model.layers[%s].mlp", CONST_INPUT_HOOK),
    "language.attention_value_output": ("language_model.model.layers[%s].self_attn.o_proj", CONST_INPUT_HOOK),
    "language.head_attention_value_output": ("language_model.model.layers[%s].self_attn.o_proj", CONST_INPUT_HOOK, (split_head_and_permute, "language.n_head")),
    "language.attention_output": ("language_model.model.layers[%s].self_attn", CONST_OUTPUT_HOOK),
    "language.attention_input": ("language_model.model.layers[%s].self_attn", CONST_INPUT_HOOK),
    "language.query_output": ("language_model.model.layers[%s].self_attn.q_proj", CONST_OUTPUT_HOOK),
    "language.key_output": ("language_model.model.layers[%s].self_attn.k_proj", CONST_OUTPUT_HOOK),
    "language.value_output": ("language_model.model.layers[%s].self_attn.v_proj", CONST_OUTPUT_HOOK),
    "language.head_query_output": ("language_model.model.layers[%s].self_attn.q_proj", CONST_OUTPUT_HOOK, (split_head_and_permute, "language.n_head")),
    "language.head_key_output": ("language_model.model.layers[%s].self_attn.k_proj", CONST_OUTPUT_HOOK, (split_head_and_permute, "language.n_kv_head")),
    "language.head_value_output": ("language_model.model.layers[%s].self_attn.v_proj", CONST_OUTPUT_HOOK, (split_head_and_permute, "language.n_kv_head")),
    "vision.block_input": ("vision_tower.encoder.layer[%s]", CONST_INPUT_HOOK),
    "vision.block_output": ("vision_tower.encoder.layer[%s]", CONST_OUTPUT_HOOK),
    "vision.mlp_activation": ("vision_tower.encoder.layer[%s].mlp.activation_fn", CONST_OUTPUT_HOOK),
    "vision.mlp_output": ("vision_tower.encoder.layer[%s].mlp", CONST_OUTPUT_HOOK),
    "vision.mlp_input": ("vision_tower.encoder.layer[%s].mlp", CONST_INPUT_HOOK),
    "vision.attention_value_output": ("vision_tower.encoder.layer[%s].attention.projection_layer", CONST_INPUT_HOOK),
    "vision.head_attention_value_output": ("vision_tower.encoder.layer[%s].attention.projection_layer", CONST_INPUT_HOOK, (split_head_and_permute, "vision.n_head")),
    "vision.attention_output": ("vision_tower.encoder.layer[%s].attention", CONST_OUTPUT_HOOK),
    "vision.attention_input": ("vision_tower.encoder.layer[%s].attention", CONST_INPUT_HOOK),
    "vision.query_output": ("vision_tower.encoder.layer[%s].attention.q_proj", CONST_OUTPUT_HOOK),
    "vision.key_output": ("vision_tower.encoder.layer[%s].attention.k_proj", CONST_OUTPUT_HOOK),
    "vision.value_output": ("vision_tower.encoder.layer[%s].attention.v_proj", CONST_OUTPUT_HOOK),
    "vision.head_query_output": ("vision_tower.encoder.layer[%s].attention.q_proj", CONST_OUTPUT_HOOK, (split_head_and_permute, "vision.n_head")),
    "vision.head_key_output": ("vision_tower.encoder.layer[%s].attention.k_proj", CONST_OUTPUT_HOOK, (split_head_and_permute, "vision.n_head")),
    "vision.head_value_output": ("vision_tower.encoder.layer[%s].attention.v_proj", CONST_OUTPUT_HOOK, (split_head_and_permute, "vision.n_head")),
}


internvl_type_to_dimension_mapping = {
    "language.n_head": ("llm_config.num_attention_heads",),
    "language.n_kv_head": ("llm_config.num_key_value_heads",),
    "language.block_input": ("llm_config.hidden_size",),
    "language.block_output": ("llm_config.hidden_size",),
    "language.mlp_activation": ("llm_config.intermediate_size",),
    "language.mlp_output": ("llm_config.hidden_size",),
    "language.mlp_input": ("llm_config.hidden_size",),
    "language.attention_value_output": ("llm_config.hidden_size",),
    "language.head_attention_value_output": ("llm_config.hidden_size/llm_config.num_attention_heads",),
    "language.attention_output": ("llm_config.hidden_size",),
    "language.attention_input": ("llm_config.hidden_size",),
    "language.query_output": ("llm_config.hidden_size",),
    "language.key_output": ("llm_config.hidden_size",),
    "language.value_output": ("llm_config.hidden_size",),
    "language.head_query_output": ("llm_config.hidden_size/llm_config.num_attention_heads",),
    "language.head_key_output": ("llm_config.hidden_size/llm_config.num_attention_heads",),
    "language.head_value_output": ("llm_config.hidden_size/llm_config.num_attention_heads",),
    "vision.n_head": ("vision_config.num_attention_heads",),
    "vision.n_kv_head": ("vision_config.num_key_value_heads",),
    "vision.block_input": ("vision_config.hidden_size",),
    "vision.block_output": ("vision_config.hidden_size",),
    "vision.mlp_activation": ("vision_config.intermediate_size",),
    "vision.mlp_output": ("vision_config.hidden_size",),
    "vision.mlp_input": ("vision_config.hidden_size",),
    "vision.attention_value_output": ("vision_config.hidden_size",),
    "vision.head_attention_value_output": ("vision_config.hidden_size/vision_config.num_attention_heads",),
    "vision.attention_output": ("vision_config.hidden_size",),
    "vision.attention_input": ("vision_config.hidden_size",),
    "vision.query_output": ("vision_config.hidden_size",),
    "vision.key_output": ("vision_config.hidden_size",),
    "vision.value_output": ("vision_config.hidden_size",),
    "vision.head_query_output": ("vision_config.hidden_size/vision_config.num_attention_heads",),
    "vision.head_key_output": ("vision_config.hidden_size/vision_config.num_attention_heads",),
    "vision.head_value_output": ("vision_config.hidden_size/vision_config.num_attention_heads",),
}

"""internvl model with LM head"""
internvl_lm_type_to_module_mapping = {}
for k, v in internvl_type_to_module_mapping.items():
    internvl_lm_type_to_module_mapping[k] = (f"model.{v[0]}", ) + v[1:]
internvl_lm_type_to_dimension_mapping = internvl_type_to_dimension_mapping

"""internvl model with classifier head"""
internvl_classifier_type_to_module_mapping = {}
for k, v in internvl_type_to_module_mapping.items():
    internvl_classifier_type_to_module_mapping[k] = (f"model.{v[0]}", ) + v[1:]
internvl_classifier_type_to_dimension_mapping = internvl_type_to_dimension_mapping

def create_internvl(
    name="OpenGVLab/InternVL3-14B", cache_dir=None, dtype=torch.bfloat16, config=None
):
    """Creates a InternVL LM model, config, and tokenizer from the given name and revision"""
    from transformers import AutoModel, AutoTokenizer, AutoConfig
    if config is None:
        config = AutoConfig.from_pretrained(name, cache_dir=cache_dir)
        internvl = AutoModel.from_pretrained(
            name,
            config=config,
            cache_dir=cache_dir,
            trust_remote_code=True,
            torch_dtype=dtype,  # save memory
        )
        tokenizer = AutoTokenizer.from_pretrained(name, cache_dir=cache_dir)
    else:
        internvl = AutoModel.from_pretrained(config)
        tokenizer = AutoTokenizer.from_pretrained(name, cache_dir=cache_dir)
    print("loaded model")
    return config, tokenizer, internvl
