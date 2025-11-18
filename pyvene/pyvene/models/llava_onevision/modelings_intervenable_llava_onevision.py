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

llava_onevision_type_to_module_mapping = {
    "block_input": ("language_layers[%s]", CONST_INPUT_HOOK),
    "block_output": ("language_layers[%s]", CONST_OUTPUT_HOOK),
    "mlp_activation": ("language_layers[%s].mlp.act_fn", CONST_OUTPUT_HOOK),
    "mlp_output": ("language_layers[%s].mlp", CONST_OUTPUT_HOOK),
    "mlp_input": ("language_layers[%s].mlp", CONST_INPUT_HOOK),
    "attention_value_output": ("language_layers[%s].self_attn.o_proj", CONST_INPUT_HOOK),
    "head_attention_value_output": ("language_layers[%s].self_attn.o_proj", CONST_INPUT_HOOK, (split_head_and_permute, "n_head")),
    "attention_output": ("language_layers[%s].self_attn", CONST_OUTPUT_HOOK),
    "attention_input": ("language_layers[%s].self_attn", CONST_INPUT_HOOK),
    "query_output": ("language_layers[%s].self_attn.q_proj", CONST_OUTPUT_HOOK),
    "key_output": ("language_layers[%s].self_attn.k_proj", CONST_OUTPUT_HOOK),
    "value_output": ("language_layers[%s].self_attn.v_proj", CONST_OUTPUT_HOOK),
    "head_query_output": ("language_layers[%s].self_attn.q_proj", CONST_OUTPUT_HOOK, (split_head_and_permute, "n_head")),
    "head_key_output": ("language_layers[%s].self_attn.k_proj", CONST_OUTPUT_HOOK, (split_head_and_permute, "n_kv_head")),
    "head_value_output": ("language_layers[%s].self_attn.v_proj", CONST_OUTPUT_HOOK, (split_head_and_permute, "n_kv_head")),
    "vision.block_input": ("vision_tower.vision_model.encoder.layers[%s]", CONST_INPUT_HOOK),
    "vision.block_output": ("vision_tower.vision_model.encoder.layers[%s]", CONST_OUTPUT_HOOK),
    "vision.mlp_activation": ("vision_tower.vision_model.encoder.layers[%s].mlp.activation_fn", CONST_OUTPUT_HOOK),
    "vision.mlp_output": ("vision_tower.vision_model.encoder.layers[%s].mlp", CONST_OUTPUT_HOOK),
    "vision.mlp_input": ("vision_tower.vision_model.encoder.layers[%s].mlp", CONST_INPUT_HOOK),
    "vision.attention_value_output": ("vision_tower.vision_model.encoder.layers[%s].self_attn.out_proj", CONST_INPUT_HOOK),
    "vision.head_attention_value_output": ("vision_tower.vision_model.encoder.layers[%s].self_attn.out_proj", CONST_INPUT_HOOK, (split_head_and_permute, "vision.n_head")),
    "vision.attention_output": ("vision_tower.vision_model.encoder.layers[%s].self_attn", CONST_OUTPUT_HOOK),
    "vision.attention_input": ("vision_tower.vision_model.encoder.layers[%s].self_attn", CONST_INPUT_HOOK),
    "vision.query_output": ("vision_tower.vision_model.encoder.layers[%s].self_attn.q_proj", CONST_OUTPUT_HOOK),
    "vision.key_output": ("vision_tower.vision_model.encoder.layers[%s].self_attn.k_proj", CONST_OUTPUT_HOOK),
    "vision.value_output": ("vision_tower.vision_model.encoder.layers[%s].self_attn.v_proj", CONST_OUTPUT_HOOK),
    "vision.head_query_output": ("vision_tower.vision_model.encoder.layers[%s].self_attn.q_proj", CONST_OUTPUT_HOOK, (split_head_and_permute, "vision.n_head")),
    "vision.head_key_output": ("vision_tower.vision_model.encoder.layers[%s].self_attn.k_proj", CONST_OUTPUT_HOOK, (split_head_and_permute, "vision.n_head")),
    "vision.head_value_output": ("vision_tower.vision_model.encoder.layers[%s].self_attn.v_proj", CONST_OUTPUT_HOOK, (split_head_and_permute, "vision.n_head")),
}


llava_onevision_type_to_dimension_mapping = {
    "language.n_head": ("text_config.num_attention_heads",),
    "language.n_kv_head": ("text_config.num_key_value_heads",),
    "language.block_input": ("text_config.hidden_size",),
    "language.block_output": ("text_config.hidden_size",),
    "language.mlp_activation": ("text_config.intermediate_size",),
    "language.mlp_output": ("text_config.hidden_size",),
    "language.mlp_input": ("text_config.hidden_size",),
    "language.attention_value_output": ("text_config.hidden_size",),
    "language.head_attention_value_output": ("text_config.hidden_size/text_config.num_attention_heads",),
    "language.attention_output": ("text_config.hidden_size",),
    "language.attention_input": ("text_config.hidden_size",),
    "language.query_output": ("text_config.hidden_size",),
    "language.key_output": ("text_config.hidden_size",),
    "language.value_output": ("text_config.hidden_size",),
    "language.head_query_output": ("text_config.hidden_size/text_config.num_attention_heads",),
    "language.head_key_output": ("text_config.hidden_size/text_config.num_attention_heads",),
    "language.head_value_output": ("text_config.hidden_size/text_config.num_attention_heads",),
    "vision.n_head": ("vision_config.num_attention_heads",),
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


"""llava model with LM head"""
llava_onevision_lm_type_to_module_mapping = {}
for k, v in llava_onevision_type_to_module_mapping.items():
    llava_onevision_lm_type_to_module_mapping[k] = (f"model.{v[0]}", ) + v[1:]


llava_onevision_lm_type_to_dimension_mapping = llava_onevision_type_to_dimension_mapping


"""llava model with classifier head"""
llava_onevision_classifier_type_to_module_mapping = {}
for k, v in llava_onevision_type_to_module_mapping.items():
    llava_onevision_classifier_type_to_module_mapping[k] = (f"model.{v[0]}", ) + v[1:]


llava_onevision_classifier_type_to_dimension_mapping = llava_onevision_type_to_dimension_mapping


def create_llava_onevision(
    name="llava-hf/llava-onevision-qwen2-7b-ov-hf", cache_dir=None, dtype=torch.bfloat16
):
    """Creates a llava onevision model, config, and processor from the given name and revision"""
    from transformers import LlavaOnevisionForConditionalGeneration, LlavaOnevisionConfig, AutoProcessor

    config = LlavaOnevisionConfig.from_pretrained(name, cache_dir=cache_dir)
    tokenizer = AutoProcessor.from_pretrained(name, use_fast=False)
    llava = LlavaOnevisionForConditionalGeneration.from_pretrained(
        name,
        config=config,
        cache_dir=cache_dir,
        torch_dtype=dtype,
    )

    print("loaded model")
    return config, tokenizer, llava

