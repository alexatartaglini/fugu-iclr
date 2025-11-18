import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.cache_utils import Cache
from typing import Optional, Union, Tuple, List


class MolmoWrapper(nn.Module):
    def __init__(self, model: AutoModelForCausalLM):
        super(MolmoWrapper, self).__init__()
        self.model = model
        self.config = model.config
        self.config.text_config = model.config
        self.config.vision_config = model.model.vision_backbone.config
        self.device = model.device

    def __call__(self, **kwargs):
        return self.model(**kwargs)

    def forward(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        attention_bias: Optional[torch.Tensor] = None,
        response_mask: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_masks: Optional[torch.Tensor] = None,
        image_input_idx: Optional[torch.Tensor] = None,
        subsegment_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        loss_masks: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        last_logits_only: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        append_last_valid_logits: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[Cache] = None
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return self.model.forward(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            attention_bias=attention_bias,
            response_mask=response_mask,
            images=images,
            image_masks=image_masks,
            image_input_idx=image_input_idx,
            subsegment_ids=subsegment_ids,
            position_ids=position_ids,
            past_key_values=past_key_values,
            labels=labels,
            loss_masks=loss_masks,
            use_cache=use_cache,
            last_logits_only=last_logits_only,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            append_last_valid_logits=append_last_valid_logits,
            return_dict=return_dict,
            cache_position=cache_position
        )
    
    def parameters(self):
        return self.model.parameters()
    
    def named_parameters(self):
        return self.model.named_parameters()
    
    def train(self, **kwargs):
        self.model.train(**kwargs)
    
    def eval(self, **kwargs):
        self.model.eval(**kwargs)

    def trace(self, key, **kwargs):
        return self.model.trace(key, **kwargs)
    
    def output(self):
        return self.model.output
    
    def load_state_dict(self, state_dict, **kwargs):
        self.model.load_state_dict(state_dict, **kwargs)

    def generate(self, **kwargs):
        return self.model.generate(**kwargs)
