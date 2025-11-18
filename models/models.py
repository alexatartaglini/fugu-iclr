from typing import Dict, List, Optional, Union
import base64
from io import BytesIO
import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    AutoModelForCausalLM,
    MllamaForConditionalGeneration,
    BlipProcessor,
    BlipForQuestionAnswering,
    ChameleonProcessor,
    ChameleonForConditionalGeneration,
    InternVLForConditionalGeneration,
    GenerationConfig,
    BitsAndBytesConfig,
    LlavaOnevisionForConditionalGeneration,
    Gemma3ForConditionalGeneration
)
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

from anthropic import Anthropic
from openai import OpenAI
from google import genai
from google.genai import types


########################################################
# Base Model Handler
########################################################


class BaseModelHandler:
    """Base class for model-specific handlers"""
    def __init__(
        self,
        model_id: str,
        cache_dir: str = None,
        token: str = None,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        device: str = "auto",
    ):
        self.model_id = model_id
        self.cache_dir = cache_dir
        self.token = open(token, "r", encoding="utf-8").read().strip() if token is None else token
        self.device = device
        
        # Configure quantization
        self.quantization_config = None
        if load_in_4bit:
            self.quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        elif load_in_8bit:
            self.quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_use_double_quant=True
            )
        
        self.load_model_and_processor()

        self.cls_token = False

    def __call__(self, **inputs) -> str:
        out = self.model(**inputs, do_sample=False, temperature=None, top_p=None, top_k=None)
        pred = torch.argmax(out.logits[:, -1, :], dim=-1)
        return self.processor.decode(pred, skip_special_tokens=True)

    def load_model_and_processor(self):
        """Load the model and processor for the specific model type.
        
        This method should be implemented by each model-specific handler to:
        1. Initialize the appropriate processor for the model
        2. Load the model weights from the specified model_id
        3. Set the model to evaluation mode
        4. Move the model to the appropriate device (CPU/GPU)
        
        The specific implementation will vary depending on the model architecture
        and requirements.
        """
        raise NotImplementedError

    def process_input(self, image: Image.Image, prompt: str, template: str = None, add_special_tokens: bool = True) -> Dict[str, torch.Tensor]:
        """Process input image and text for model inference.
        
        Args:
            image: PIL Image object to process
            prompt: Text prompt to process
            template: Optional template string to append to the processed text
            
        Returns:
            Dictionary containing processed model inputs, typically including:
                - input_ids: Tokenized text input
                - attention_mask: Attention mask for text
                - pixel_values: Preprocessed image tensors
                - Other model-specific inputs
        """
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "source": image},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        if image is None:
            del messages[0]["content"][0]
        
        text = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            add_special_tokens=add_special_tokens
        )
        inputs = self.processor(
            images=image,
            text=text,
            return_tensors="pt",
        )

        if template is not None:
            # Decode input_ids to text
            input_text = self.processor.tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=False)
            # Append template to text
            combined_text = input_text + template
            # Process combined text without special tokens

            inputs = self.processor(
                images=image,
                text=combined_text,
                return_tensors="pt",
                add_special_tokens=False
            )
        
        return {k: v.to(device=self.model.device) for k, v in inputs.items()}

    @torch.no_grad()
    def generate(
        self, 
        **inputs
    ) -> str:
        """Generate text from inputs.
        
        Args:
            **inputs: Model inputs including input_ids and attention_mask
            
        Returns:
            str: Generated text
        """
        generation_params = {
            'max_new_tokens': 1000,
            'do_sample': False,
            'temperature': None,
            'top_p': None
        }
        
        outputs = self.model.generate(
            **inputs,
            **generation_params
        )
        
        # Get only the newly generated tokens
        generated_tokens = outputs[0, inputs['input_ids'].size(1):]
        return self.processor.decode(generated_tokens, skip_special_tokens=True)
    
    def _get_nested_attr(self, obj, attr: str):
        """Helper to get potentially nested attributes"""
        for part in attr.split('.'):
            obj = getattr(obj, part)
        return obj
    
    def get_layers(self, modality: str) -> List[int]:
        """Get list of layers for the model."""
        raise NotImplementedError


########################################################
# Specific Model Handlers
########################################################


class LlamaHandler(BaseModelHandler):
    """Handler for LLaMA 3.2"""
    def load_model_and_processor(self):
        self.processor = AutoProcessor.from_pretrained(
            self.model_id,
            token=self.token
        )

        # Configure quantization parameters
        model_kwargs = {
            "torch_dtype": torch.bfloat16,
            "device_map": self.device,
            "cache_dir": self.cache_dir,
            "token": self.token,
        }

        if self.quantization_config:
            model_kwargs["quantization_config"] = self.quantization_config

        self.model = MllamaForConditionalGeneration.from_pretrained(
            self.model_id,
            **model_kwargs
        )
        self.model = self.model.eval()

        self.cls_token = True
    
    def get_layers(self, modality: str) -> List[int]:
        if modality == 'vision':
            return list(range(self.model.vision_model.config.num_hidden_layers))
        elif modality == 'language':
            return list(range(self.model.language_model.config.num_hidden_layers))
        elif modality == 'connector':
            return [0]
        else:
            raise ValueError(f"Invalid modality: {modality}")
        
    def get_image_resolution(self) -> int:
        try:
            return self.model.vision_model.config.image_size[0]
        except:
            return 560
        

class LlavaOnevisionHandler(BaseModelHandler):
    """Handler for Llava Onevision"""
    def load_model_and_processor(self):
        self.processor = AutoProcessor.from_pretrained(
            self.model_id,
            token=self.token
        )

        # Configure quantization parameters
        model_kwargs = {
            "torch_dtype": torch.bfloat16,
            "device_map": self.device,
            "cache_dir": self.cache_dir,
            "token": self.token,
        }

        if self.quantization_config:
            model_kwargs["quantization_config"] = self.quantization_config

        self.model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            self.model_id,
            **model_kwargs
        )
        self.model = self.model.eval()

        self.cls_token = False

    def process_input(self, image: Image.Image, prompt: str, template: str = None, add_special_tokens: bool = True) -> Dict[str, torch.Tensor]:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            },
        ]
        if image is None:
            del messages[0]["content"][0]
        
        text = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            add_special_tokens=add_special_tokens
        )

        inputs = self.processor(
            images=image,
            text=text,
            return_tensors="pt",
        )

        if template is not None:
            # Decode input_ids to text
            input_text = self.processor.tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=False)
            input_text = input_text.replace("<image>", "")
            input_text = input_text.replace("<|im_start|>user", "<|im_start|>user <image>")

            # Append template to text
            combined_text = input_text + template

            # Process combined text without special tokens
            inputs = self.processor(
                images=image,
                text=combined_text,
                return_tensors="pt",
                add_special_tokens=False
            )
        
        return {k: v.to(device=self.model.device) for k, v in inputs.items()}
    
    def get_layers(self, modality: str) -> List[int]:
        if modality == 'vision':
            return list(range(self.model.model.config.vision_config.num_hidden_layers))
        elif modality == 'language':
            return list(range(self.model.model.config.text_config.num_hidden_layers))
        else:
            raise ValueError(f"Invalid modality: {modality}")
        
    def get_image_resolution(self) -> int:
        return self.model.config.vision_config.image_size  #- 6  # 6 pixels are removed during patchification
        

class ChameleonHandler(BaseModelHandler):
    """Handler for Chameleon"""
    def load_model_and_processor(self):
        self.processor = ChameleonProcessor.from_pretrained(
            self.model_id,
            token=self.token
        )

        model_kwargs = {
            "device_map": self.device,
            "cache_dir": self.cache_dir,
            "token": self.token,
        }
        
        if self.quantization_config:
            model_kwargs["quantization_config"] = self.quantization_config
        
        self.model = ChameleonForConditionalGeneration.from_pretrained(
            self.model_id,
            **model_kwargs
        )
        self.model = self.model.eval()

    def process_input(self, image: Image.Image, prompt: str, template: str = None, add_special_tokens: bool = True) -> Dict[str, torch.Tensor]:
        # Chameleon expects "<image>" token in the prompt
        if "<image>" not in prompt and template is None:
            prompt = prompt + "<image>"
        inputs = self.processor(
            images=image,
            text=prompt,
            return_tensors="pt",
            add_special_tokens=add_special_tokens
        )

        if template is not None:
            # Decode input_ids to text
            input_text = self.processor.tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=False)
            # Append template to text
            combined_text = input_text + template + "<image>"
            # Process combined text without special tokens
            inputs = self.processor(
                images=image,
                text=combined_text,
                return_tensors="pt",
                add_special_tokens=False
            )
        
        return {k: v.to(device=self.model.device) for k, v in inputs.items()}
    
    def get_layers(self, modality: str) -> List[int]:
        return list(range(self.model.model.config.num_hidden_layers))


class MolmoHandler(BaseModelHandler):
    """Handler for Molmo"""
    def __call__(self, **inputs) -> str:
        out = self.model(**inputs)
        pred = torch.argmax(out.logits[:, -1, :], dim=-1)
        return self.processor.tokenizer.decode(pred, skip_special_tokens=True)
    
    def load_model_and_processor(self):
        self.processor = AutoProcessor.from_pretrained(
            self.model_id,
            trust_remote_code=True,
        )

        model_kwargs = {
            "trust_remote_code": True,
            "device_map": self.device,
            "cache_dir": self.cache_dir,
            "torch_dtype": torch.float32,  # Molmo needs float32
        }

        if self.quantization_config:
            # Override compute dtype for Molmo
            if hasattr(self.quantization_config, "bnb_4bit_compute_dtype"):
                self.quantization_config.bnb_4bit_compute_dtype = torch.float32
            model_kwargs["quantization_config"] = self.quantization_config
            
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            **model_kwargs
        )
        self.model = self.model.eval()

    def process_input(self, image: Image.Image, prompt: str, template: str = None, add_special_tokens: bool = True) -> Dict[str, torch.Tensor]:
        inputs = self.processor.process(
            images=[image] if image else None,
            text=prompt,
            return_tensors="pt",
            add_special_tokens=add_special_tokens
        )

        if template is not None:
            # Encode template without special tokens
            template_ids = self.processor.tokenizer.encode(template, add_special_tokens=False)
            # Append template tokens to input_ids
            inputs['input_ids'] = torch.cat([
                inputs['input_ids'],
                torch.tensor(template_ids, device=inputs['input_ids'].device)
            ], dim=0)

        # Molmo expects specific input formatting
        return {k: v.to(device=self.model.device).unsqueeze(0) for k, v in inputs.items()}

    @torch.no_grad()
    def generate(self, **inputs) -> str:
        """Override for Molmo's specific generation method"""
        outputs = self.model.generate_from_batch(
            inputs,
            GenerationConfig(
                max_new_tokens=1000,
                stop_strings="<|endoftext|>",
                do_sample=False,
                temperature=None,
                top_p=None
            ),
            tokenizer=self.processor.tokenizer
        )
        generated_tokens = outputs[0, inputs['input_ids'].size(1):]
        return self.processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
    def get_layers(self, modality: str) -> List[int]:
        if modality == 'vision':
            return list(range(self.model.model.vision_backbone.config.n_layers))
        elif modality == 'language':
            return list(range(self.model.model.config.text_layers))
        else:
            raise ValueError(f"Invalid modality: {modality}")
        

class InternVLHandler(BaseModelHandler):
    """Handler for InternVL"""
    def __call__(self, **inputs) -> str:
        prompt = self.processor(inputs["prompt"], add_special_tokens=False)
        response = self.model(pixel_values=inputs["pixel_values"], **prompt)
        return response
    
    def load_model_and_processor(self):
        self.processor = AutoProcessor.from_pretrained(self.model_id)

        model_kwargs = {
            "trust_remote_code": True,
            "device_map": self.device,
            "cache_dir": self.cache_dir,
            "torch_dtype": torch.bfloat16,
            "load_in_8bit": False,
        }

        if self.quantization_config:
            model_kwargs["quantization_config"] = self.quantization_config

        self.model = InternVLForConditionalGeneration.from_pretrained(
            self.model_id,
            **model_kwargs
        )
        self.model = self.model.eval()

        self.cls_token = True

    def process_input(self, image: Image.Image, prompt: str, template: str = None, add_special_tokens: bool = True) -> Dict[str, torch.Tensor]:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "source": image},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        if image is None:
            del messages[0]["content"][0]
        
        text = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            add_special_tokens=add_special_tokens
        )
        inputs = self.processor(
            images=image,
            text=text,
            return_tensors="pt",
        )

        if template is not None:
            # Decode input_ids to text
            input_text = self.processor.tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=False)
            input_text = input_text.replace("<IMG_CONTEXT>", "").replace("<img>", "<IMG_CONTEXT>").replace("</img>", "")
            # Append template to text
            combined_text = input_text + template
            # Process combined text without special tokens

            inputs = self.processor(
                images=image,
                text=combined_text,
                return_tensors="pt",
                add_special_tokens=False
            )

            input_text = self.processor.tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=False)
        
        inputs = {k: v.to(device=self.model.device) for k, v in inputs.items()}
        for k, v in inputs.items():
            if v.dtype in (torch.float16, torch.float32, torch.bfloat16):
                inputs[k] = v.to(dtype=self.model.dtype)
        return inputs
    
    def get_layers(self, modality: str) -> List[int]:
        if modality == 'vision':
            return list(range(self.model.vision_tower.config.num_hidden_layers))
        elif modality == 'language':
            return list(range(self.model.language_model.config.num_hidden_layers))
        elif modality == 'connector':
            return [0]
        else:
            raise ValueError(f"Invalid modality: {modality}")
        
    def get_image_resolution(self) -> int:
        return self.model.vision_tower.config.image_size[0]


class BlipHandler(BaseModelHandler):
    """Handler for BLIP2"""
    def load_model_and_processor(self):
        self.processor = BlipProcessor.from_pretrained(self.model_id)
        self.model = BlipForQuestionAnswering.from_pretrained(
            self.model_id,
            cache_dir=self.cache_dir,
            torch_dtype=torch.bfloat16
        )
        self.model = self.model.eval()
    
    def process_input(self, image: Image.Image, prompt: str, template: str = None, add_special_tokens: bool = True) -> Dict[str, torch.Tensor]:
        inputs = self.processor(
            images=image,
            text=prompt,
            return_tensors="pt",
            add_special_tokens=add_special_tokens
        )

        if template is not None:
            # Decode input_ids to text
            input_text = self.processor.tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=False)
            # Append template to text
            combined_text = input_text + template
            # Process combined text without special tokens
            inputs = self.processor(
                images=image,
                text=combined_text,
                return_tensors="pt",
                add_special_tokens=False
            )
        
        return {k: v.to(device=self.model.device) for k, v in inputs.items()}

    @torch.no_grad()
    def generate(self, **inputs) -> str:
        """Override for BLIP's specific output handling"""
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=1000,
            do_sample=False,
            temperature=None,
            top_p=None,
        )
        # BLIP doesn't need token slicing
        return self.processor.decode(outputs[0], skip_special_tokens=True)


class ClaudeHandler(BaseModelHandler):
    """Handler for Claude"""
    def load_model_and_processor(self):
        self.client = Anthropic(api_key=self.token)
    
    def process_input(self, image: Image.Image, prompt: str, template: str = None, add_special_tokens: bool = True) -> Dict[str, torch.Tensor]:
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        image = base64.b64encode(buffered.getvalue()).decode("utf-8")

        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": image,
                        },
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ]
        return {"messages": messages}

    def generate(self, **inputs) -> str:
        response = self.client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=1000,
            temperature=0,
            messages=inputs["messages"],
        )
        return response.content[0].text
    
    def get_layers(self, modality: str) -> List[int]:
        return


class ChatGPTHandler(BaseModelHandler):
    """Handler for ChatGPT"""
    def load_model_and_processor(self):
        self.client = OpenAI(api_key=self.token)
    
    def process_input(self, image: Image.Image, prompt: str, template: str = None, add_special_tokens: bool = True) -> Dict[str, torch.Tensor]:
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        image = base64.b64encode(buffered.getvalue()).decode("utf-8")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image}",
                        },
                    },
                ]
            }
        ]

        return {"messages": messages}

    def generate(self, **inputs) -> str:
        if self.model_id == "gpt-5":
            response = self.client.chat.completions.create(
                model=self.model_id,
                max_completion_tokens=1000,
                messages=inputs["messages"],
            )
        else:
            response = self.client.chat.completions.create(
                model=self.model_id,
                max_tokens=1000,
                temperature=0,
                messages=inputs["messages"],
            )
        return response.choices[0].message.content
    
    def get_layers(self, modality: str) -> List[int]:
        return


class GeminiHandler(BaseModelHandler):
    """Handler for ChatGPT"""
    def load_model_and_processor(self):
        self.client = genai.Client(api_key=self.token)
    
    def process_input(self, image: str, prompt: str, template: str = None, add_special_tokens: bool = True) -> Dict[str, torch.Tensor]:
        with open(image, 'rb') as f:
            image_bytes = f.read()

        contents=[
            types.Part.from_bytes(
                data=image_bytes,
                mime_type='image/png',
            ),
            prompt
        ]

        return {"contents": contents}

    def generate(self, **inputs) -> str:
        response = self.client.models.generate_content(
            model=self.model_id,
            contents=inputs["contents"],
        )
        return response.text
    
    def get_layers(self, modality: str) -> List[int]:
        return


class GemmaHandler(BaseModelHandler):
    """Handler for Gemma"""
    def load_model_and_processor(self):
        model_kwargs = {
            "trust_remote_code": True,
            "device_map": self.device,
            "cache_dir": self.cache_dir,
            "torch_dtype": torch.bfloat16,
            "load_in_8bit": False,
        }

        if self.quantization_config:
            model_kwargs["quantization_config"] = self.quantization_config

        self.model = Gemma3ForConditionalGeneration.from_pretrained(
            self.model_id, **model_kwargs
        ).eval()

        self.processor = AutoProcessor.from_pretrained(self.model_id)
    
    def get_layers(self, modality: str) -> List[int]:
        return


def get_model_handler(model_name: str, **kwargs) -> BaseModelHandler:
    """Factory function to get the appropriate model handler"""
    handlers = {
        "llama": LlamaHandler,
        "chameleon": ChameleonHandler,
        "molmo": MolmoHandler,
        "blip": BlipHandler,
        "claude": ClaudeHandler,
        "gpt": ChatGPTHandler,
        "internvl": InternVLHandler,
        "llava-onevision": LlavaOnevisionHandler,
        "gemma": GemmaHandler,
        "gemini": GeminiHandler
    }
    
    model_type = model_name.lower().split("/")[-1]
    for key, handler in handlers.items():
        if key in model_type:
            return handler(model_name, **kwargs)
    
    raise ValueError(f"Unknown model type: {model_name}")


def get_available_models() -> List[str]:
    """Get a list of all available model handlers"""
    return [
        "meta-llama/Llama-3.2-11B-Vision-Instruct",
        "Salesforce/blip-vqa-base",
        "allenai/Molmo-7B-D-0924",
        "allenai/Molmo-7B-O-0924",
        "facebook/chameleon-7b",
        "claude-3-5-sonnet",
        "claude-3-5-haiku",
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-5",
        "OpenGVLab/InternVL3-8B",
        "OpenGVLab/InternVL2_5-8B",
        "OpenGVLab/InternVL3-14B-hf",
        "llava-hf/llava-onevision-qwen2-7b-ov-hf",
        "google/gemma-3-12b-it",
        "gemini-2.5-pro",
        "gemini-2.5-flash"
    ]
