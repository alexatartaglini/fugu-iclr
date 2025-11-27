import os
import time
from typing import Union, List, Tuple
import re
import argparse
import json
import tqdm
import numpy as np
import pandas as pd
from PIL import Image
import torch

from models.models import get_model_handler
from utils.cot_templates import get_cot_template
from utils.generate_prompts import PROMPT_PREFIX


def prepare_for_storage(result_dict):
    """Convert complex data types to JSON strings for nested structures"""
    processed = {}
    def convert_numpy_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        return obj

    for key, value in result_dict.items():
        processed[key] = convert_numpy_types(value)
    return processed

def extract_numerical_answers(text: str) -> Union[List[float], List[Tuple[float, float]]]:
    """Using a regex to extract numerical answers from the text. Used for validating LLM judge outputs (analyze_results.py)"""

    # Dictionary for converting word numbers to digits
    word_to_num = {
        'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4,
        'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9,
        'ten': 10, 'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14,
        'fifteen': 15, 'sixteen': 16, 'seventeen': 17, 'eighteen': 18, 'nineteen': 19,
        'twenty': 20, 'thirty': 30, 'forty': 40, 'fifty': 50, 'sixty': 60,
        'seventy': 70, 'eighty': 80, 'ninety': 90, 'hundred': 100,
        'thousand': 1000, 'million': 1000000, 'billion': 1000000000
    }
    
    text = text.lower()
    
    # Find all coordinate pairs
    coord_pairs = re.findall(r'\(([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\)', text)
    if coord_pairs:
        # Return list of tuples for coordinates
        return [(float(x), float(y)) for x, y in coord_pairs]
    
    # If no coordinates found, look for other number formats:
    answers = []
    
    # Find standalone numbers
    standalone_nums = re.findall(r'[-+]?\d*\.?\d+', text)
    answers.extend(float(num) for num in standalone_nums)
    
    # Find word numbers
    word_nums = re.findall(r'\b(' + '|'.join(word_to_num.keys()) + r')\b', text)
    answers.extend(word_to_num[word] for word in word_nums)
    
    # Find lists
    lists = re.findall(r'\[([^\]]+)\]', text)
    for list_str in lists:
        list_nums = re.findall(r'[-+]?\d*\.?\d+', list_str)
        answers.extend(float(num) for num in list_nums)
    
    return answers

def get_behavioral_results(
    model_id,
    stimuli_dir,
    save_dir,
    cot_type=None,
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    """Get behavioral results for a model on dot stimuli."""
    
    dataset = pd.read_csv(f"{stimuli_dir}/stimuli_and_prompts.csv")
        
    # Load model
    model = get_model_handler(model_id, device=device)
    
    # Create save directory
    save_dir = f"{save_dir}/{model_id.split('/')[-1]}/{stimuli_dir.split('/')[-1]}"
    os.makedirs(save_dir, exist_ok=True)

    # Pre load metadatas and images
    metadata_files = {
        row['id']: np.load(f"{stimuli_dir}/metadata/{row['id']}.npy", allow_pickle=True).item() for _, row in dataset.iterrows()
    }
    images = {row['id']: Image.open(f"{stimuli_dir}/stimuli/{row['id']}.png") for _, row in dataset.iterrows()}

    results = []
    for _, row in tqdm.tqdm(dataset.iterrows(), desc="Running behavioral eval...", total=len(dataset)):
        stimulus_results_dir = f"{save_dir}/{row['id']}_{row['task_type']}{f'_{cot_type}' if cot_type else ''}.json"
        if os.path.exists(stimulus_results_dir):  # Skip if results already exist
            continue

        # Load metadata and image
        metadata = metadata_files[row['id']]
        if "gemini" in model_id:  # gemini needs to load image from file
            image = f"{stimuli_dir}/stimuli/{row['id']}.png"
        else:
            image = images[row['id']]

        # Get prompt and template
        prompt = f"{PROMPT_PREFIX}\n{row['prompt']}"
        template = get_cot_template(model, row, cot_type=cot_type, root_dir=stimuli_dir)

        # Process inputs
        if cot_type == "text_only":
            inputs = model.process_input(None, prompt, template=template)
        else:
            inputs = model.process_input(image, prompt, template=template)
        
        # Get model prediction
        generated_text = model.generate(**inputs)
        
        # Extract numerical answer(s)
        extracted_answer = extract_numerical_answers(generated_text)
        
        # Store results
        result = {
            "model_id": model_id,
            "id": metadata["id"],
            "generation_type": cot_type if cot_type else "none",
            "full_response": generated_text,
            "extracted_answer": extracted_answer,
            "template": template if template else "none",
            **row.to_dict(),
            **metadata,
        }
        results.append(result)
            
        # Save individual stimulus results
        with open(stimulus_results_dir, 'w', encoding='utf-8') as f:
            try:
                json.dump(prepare_for_storage(result), f, indent=2)
            except Exception as e:
                print(f"Error saving result for {row['id']}: {e}")
                for key, value in result.items():
                    print(f"{key}: {type(value)}")
                    if isinstance(value, dict):
                        for subkey, subvalue in value.items():
                            print(f"  {subkey}: {type(subvalue)}")   
    
    return

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, required=True, 
                       help="HuggingFace model ID",
                       choices=["meta-llama/Llama-3.2-11B-Vision-Instruct",
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
                                "gemini-2.5-flash"])
    parser.add_argument("--stimuli_dir", type=str,
                       default="datasets/bar_chart/bar_charts",
                       help="Directory containing stimuli and metadata")
    parser.add_argument("--save_dir", type=str,
                       default="results",
                       help="Directory to save results")
    parser.add_argument("--cot_type", type=str, default=None,
                       help="Type of COT to use")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to run model on")
    
    args = parser.parse_args()
    
    get_behavioral_results(
        model_id=args.model_id,
        stimuli_dir=args.stimuli_dir,
        save_dir=args.save_dir,
        cot_type=args.cot_type,
        device=args.device,
    )

if __name__ == "__main__":
    main()