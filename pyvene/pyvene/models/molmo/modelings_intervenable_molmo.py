"""
Each modeling file in this library is a mapping between
abstract naming of intervention anchor points and actual
model module defined in the huggingface library.

We also want to let the intervention library know how to
config the dimensions of intervention based on model config
defined in the huggingface library.

Author: Alexa Tartaglini
"""


import torch
from ..constants import *


# For now, there is only support for Molmo-D (Qwen backbone)
molmo_type_to_module_mapping = {
    "language.block_input": ("model.model.transformer.blocks[%s]", CONST_INPUT_HOOK),
    "language.block_output": ("model.model.transformer.blocks[%s]", CONST_OUTPUT_HOOK),
    "language.mlp_activation": ("model.model.transformer.blocks[%s].act", CONST_OUTPUT_HOOK),
    "language.mlp_output": ("model.model.transformer.blocks[%s].ff_out", CONST_OUTPUT_HOOK),
    "language.mlp_input": ("model.model.transformer.blocks[%s].ff_out", CONST_INPUT_HOOK),
    "language.attention_value_output": ("model.model.transformer.blocks[%s].attn_out", CONST_INPUT_HOOK),
    "language.head_attention_value_output": ("model.model.transformer.blocks[%s].attn_out", CONST_INPUT_HOOK, (split_head_and_permute, "language.n_head")),
    "language.attention_output": ("model.model.transformer.blocks[%s].attn_out", CONST_OUTPUT_HOOK),
    "language.attention_input": ("model.model.transformer.blocks[%s].attn_out", CONST_INPUT_HOOK),
    #"query_output": ("layers[%s].self_attn.q_proj", CONST_OUTPUT_HOOK),
    #"key_output": ("layers[%s].self_attn.k_proj", CONST_OUTPUT_HOOK),
    #"value_output": ("layers[%s].self_attn.v_proj", CONST_OUTPUT_HOOK),
    #"head_query_output": ("layers[%s].self_attn.q_proj", CONST_OUTPUT_HOOK, (split_head_and_permute, "n_head")),
    #"head_key_output": ("layers[%s].self_attn.k_proj", CONST_OUTPUT_HOOK, (split_head_and_permute, "n_kv_head")),
    #"head_value_output": ("layers[%s].self_attn.v_proj", CONST_OUTPUT_HOOK, (split_head_and_permute, "n_kv_head")),
    "vision.block_input": ("model.model.vision_backbone.image_vit.transformer.resblocks[%s]", CONST_INPUT_HOOK),
    "vision.block_output": ("model.model.vision_backbone.image_vit.transformer.resblocks[%s]", CONST_OUTPUT_HOOK),
    "vision.mlp_activation": ("model.model.vision_backbone.image_vit.transformer.resblocks[%s].feed_forward.act", CONST_OUTPUT_HOOK),
    "vision.mlp_output": ("model.model.vision_backbone.image_vit.transformer.resblocks[%s].feed_forward", CONST_OUTPUT_HOOK),
    "vision.mlp_input": ("model.model.vision_backbone.image_vit.transformer.resblocks[%s].feed_forward", CONST_INPUT_HOOK),
    "vision.attention_value_output": ("model.model.vision_backbone.image_vit.transformer.resblocks[%s].attention.wo", CONST_INPUT_HOOK),
    "vision.head_attention_value_output": ("model.model.vision_backbone.image_vit.transformer.resblocks[%s].attention.wo", CONST_INPUT_HOOK, (split_head_and_permute, "vision.n_head")),
    "vision.attention_output": ("model.model.vision_backbone.image_vit.transformer.resblocks[%s].attention", CONST_OUTPUT_HOOK),
    "vision.attention_input": ("model.model.vision_backbone.image_vit.transformer.resblocks[%s].attention", CONST_INPUT_HOOK),
    "vision.query_output": ("model.model.vision_backbone.image_vit.transformer.resblocks[%s].attention.wq", CONST_OUTPUT_HOOK),
    "vision.key_output": ("model.model.vision_backbone.image_vit.transformer.resblocks[%s].attention.wk", CONST_OUTPUT_HOOK),
    "vision.value_output": ("model.model.vision_backbone.image_vit.transformer.resblocks[%s].attention.wv", CONST_OUTPUT_HOOK),
    "vision.head_query_output": ("model.model.vision_backbone.image_vit.transformer.resblocks[%s].attention.wq", CONST_OUTPUT_HOOK, (split_head_and_permute, "vision.n_head")),
    "vision.head_key_output": ("model.model.vision_backbone.image_vit.transformer.resblocks[%s].attention.wk", CONST_OUTPUT_HOOK, (split_head_and_permute, "vision.n_head")),
    "vision.head_value_output": ("model.model.vision_backbone.image_vit.transformer.resblocks[%s].attention.wv", CONST_OUTPUT_HOOK, (split_head_and_permute, "vision.n_head")),
}


molmo_type_to_dimension_mapping = {
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
    "vision.n_head": ("vision_config.n_heads",),
    "vision.n_kv_head": ("vision_config.n_kv_heads",),
    "vision.block_input": ("vision_config.d_model",),
    "vision.block_output": ("vision_config.d_model",),
    "vision.mlp_activation": ("vision_config.mlp_hidden_size",),
    "vision.mlp_output": ("vision_config.d_model",),
    "vision.mlp_input": ("vision_config.d_model",),
    "vision.attention_value_output": ("vision_config.d_model",),
    "vision.head_attention_value_output": ("vision_config.d_model/vision_config.n_heads",),
    "vision.attention_output": ("vision_config.d_model",),
    "vision.attention_input": ("vision_config.d_model",),
    "vision.query_output": ("vision_config.d_model",),
    "vision.key_output": ("vision_config.d_model",),
    "vision.value_output": ("vision_config.d_model",),
    "vision.head_query_output": ("vision_config.d_model/vision_config.n_heads",),
    "vision.head_key_output": ("vision_config.d_model/vision_config.n_heads",),
    "vision.head_value_output": ("vision_config.d_model/vision_config.n_heads",),
}


"""molmo model with LM head"""
molmo_lm_type_to_module_mapping = {}
for k, v in molmo_type_to_module_mapping.items():
    molmo_lm_type_to_module_mapping[k] = (f"model.{v[0]}", ) + v[1:]


molmo_lm_type_to_dimension_mapping = molmo_type_to_dimension_mapping


"""molmo model with classifier head"""
molmo_classifier_type_to_module_mapping = {}
for k, v in molmo_type_to_module_mapping.items():
    molmo_classifier_type_to_module_mapping[k] = (f"model.{v[0]}", ) + v[1:]


molmo_classifier_type_to_dimension_mapping = molmo_type_to_dimension_mapping


def create_molmo(
    name="allenai/Molmo-7B-D-0924", cache_dir=None, dtype=torch.bfloat16, config=None,
    revision='main'
):
    """Creates a Molmo model, config, and tokenizer from the given name and revision"""
    from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor
    if config is None:
        config = AutoConfig.from_pretrained(name, cache_dir=cache_dir)
        molmo = AutoModelForCausalLM.from_pretrained(
            name,
            config=config,
            cache_dir=cache_dir,
            torch_dtype=dtype,
            revision=revision,
        )
        tokenizer = AutoProcessor.from_pretrained(name, cache_dir=cache_dir, trust_remote_code=True)
    else:
        molmo = AutoModelForCausalLM(config, cache_dir=cache_dir, revision=revision)
        tokenizer = AutoProcessor.from_pretrained(name, cache_dir=cache_dir, trust_remode_code=True)
    print("loaded model")
    return config, tokenizer, molmo
