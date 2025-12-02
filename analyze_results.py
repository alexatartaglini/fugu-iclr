from typing import TypedDict, Optional, Dict, Any
import os
import tqdm
import argparse

import pandas as pd
import numpy as np
import json
import glob

from anthropic import Anthropic


client = Anthropic(api_key=open("anthropic_api_key.txt", "r").read().strip())

SYMBOL_TO_SHAPE_MAP = {
    "o": "circle",
    "s": "square",
    "^": "triangle",
    "*": "star"
}

task_type_to_answer_type = {
    "count": "should be a natural number",
    "position": "should be formatted as an ordered pair enclosed within parentheses, where both values are whole numbers",
    "distance": "should be a non-negative integer",
    "min_x": "should be a simple two-word noun phrase consisting of a modifier (color word) and noun (shape word)",
    "min_y": "should be a simple two-word noun phrase consisting of a modifier (color word) and noun (shape word)",
    "max_x": "should be a simple two-word noun phrase consisting of a modifier (color word) and noun (shape word)",
    "max_y": "should be a simple two-word noun phrase consisting of a modifier (color word) and noun (shape word)",
    "mean": "should be formatted as an ordered pair enclosed within parentheses, where both values are whole numbers",
    "correlation": "should be the word 'higher' or 'lower' (or a synonym thereof)",
    "cluster": "should be a color word, possibly modifying the word 'cluster' (e.g. 'blue cluster', 'red cluster', etc.)",
    "function": "should be 'linear', 'quadratic', 'exponential', or 'logarithmic' (or a synonym thereof)",
    "outlier": "should be formatted as an ordered pair enclosed within parentheses, where both values are whole numbers",
}
task_type_to_answer_type_bar = {
    "count": "should be a natural number",
    "position": "should be a natural number",
    "distance": "should be a non-negative integer",
    "min_x": "should be a simple color word (possibly modifying the word 'bar' or paired with an alphabetical letter giving the category)",
    "min_y": "should be a simple color word (possibly modifying the word 'bar' or paired with an alphabetical letter giving the category)",
    "max_x": "should be a simple color word (possibly modifying the word 'bar' or paired with an alphabetical letter giving the category)",
    "max_y": "should be a simple color word (possibly modifying the word 'bar' or paired with an alphabetical letter giving the category)",
    "mean": "should be a natural number",
}
task_type_to_answer_type_line = {
    "count": "should be a natural number",
    "position": "should either be a natural number or an xy coordinate (where the y value is the answer)",
    "distance": "should be a non-negative integer",
    "min_x": "should be an integer",
    "min_y": "should be an integer",
    "max_x": "should be an integer",
    "max_y": "should be an integer",
    "mean": "should be a natural number",
}

DEFAULT_VALIDATION_PROMPT = """I have asked a vision-language model to answer a {task_type} question about a plot. Here is the question:

"{question}"

The ground truth answer (or possible answers, formatted as a list) to the question are: {answer}.
The model is correct if it returns an answer that is in the set of possible answers.

This is the full response from the model:

"{full_response}"

Based on the type of task, I expect that the answer {answer_spec}.
The model may have responded with an answer that is not fully formatted correctly.
For example, it may answer "the star symbol" instead of specifying a color.
It may also omit the first parenthesis in a parenthesis pair.
Finally, the model may have forgotten to round the numbers to the nearest integer.
If this answer matches one of the possible answers, it is still correct.

Please analyze the model's response and extract an answer that matches the expected format above.
Try to be as faithful as possible to the model's response while still matching the expected format.
Numbers may be provided as words in the response, and you should convert them to numbers.
If an answer is able to be extracted, please also analyze if it is correct by comparing it to the ground truth answer(s).
If no good answer can be found, please explain why.

Return your analysis in this format:
Extracted Answer: [number or 'None']
Correct: [True/False]
Explanation (if 'None'): [your reasoning]"""

CHARXIV_VALIDATION_PROMPT = """I have asked a vision-language model to answer a question about a plot. Here is the question:

"{question}"

The ground truth answer to the question is: "{answer}".
This is the model's full response: "{full_response}"

Please analyze the model's response, extract an answer, and judge if it is correct. 
The model may have responded with an answer that is not fully formatted correctly.
* Give correct = True if and only if the extracted answer and the ground truth answer are referring to the same term. It's acceptable to have different grammar or form (e.g., α and alpha. It's acceptable to omit letter prefixes (e.g., (a) Increment over time and Increment over time).
* Give correct = False if any term in the extracted answer is different from the ground truth answer.
* When ground truth answer is "Not Applicable", the response must express "Not Applicable" to receive a correct = True.

Return your analysis in this format:
Extracted Answer: [the model's answer, extracted from the full response]
Correct: [True/False]
Explanation: [your reasoning]"""

cot_analysis_prompt = """Determine whether this chain-of-thought response lists specific data points as part of its reasoning (not merely as the final answer) for a question about a plot:

"{cot_response}"

Ground truth information about the image:
- Positions: {ground_truth_points}
- Type of chart: {ground_truth_chart_type}

Definition of "listing points":
- The response explicitly names data points as an intermediate reasoning step (e.g., "points at (3, 5) and (7, 2)", "bars at 4 and 6").
- Data points can be (x, y) coordinates or numeric values that correspond to specific chart elements like scatter points, bars, or markers.
- Do NOT count a single value only appearing as the final answer. Only count listings used while working through the problem.
- Arithmetic results alone do not count; we only care about explicit listings of points.

Return JSON in this format:
    "lists_points": boolean,  # True if points are listed as part of intermediate reasoning, else False
    "explanation": "Brief justification for the decision."
"""

class ValidationResult(TypedDict):
    correct: Optional[bool]
    extracted_answer: Optional[str]
    explanation: Optional[str]

    def to_dict(self):
        return {
            "correct": self["correct"],
            "extracted_answer": self["extracted_answer"],
            "answer_validation_explanation": self["explanation"]
        }

class COTAnalysisResult(TypedDict):
    lists_points: bool
    explanation: str

    def to_dict(self):
        return {
            "cot_lists_points": self["lists_points"],
            "cot_analysis_explanation": self["explanation"]
        }
    
def analyze_cot_with_llm(cot_response, ground_truth_points, ground_truth_chart_type, judge="claude-sonnet-4-5"):
    """
    Analyze whether a chain-of-thought response lists data points as part of its reasoning.
    Returns a COTAnalysisResult.
    """
    prompt = cot_analysis_prompt.format(
        cot_response=cot_response,
        ground_truth_points=ground_truth_points,
        ground_truth_chart_type=ground_truth_chart_type,
    )

    response = client.messages.create(
        model=judge,
        max_tokens=1000,
        temperature=0,
        messages=[{
            "role": "user",
            "content": prompt
        }]
    )

    try:
        return json.loads(response.content[0].text)
    except json.JSONDecodeError:
        return {
            "error": "Failed to parse LLM response",
            "raw_response": response.content[0].text
        }

def validate_extraction_with_llm(task_type, question, answer, full_response, validation_prompt=DEFAULT_VALIDATION_PROMPT, judge="claude-sonnet-4-5"):
    """
    Use an LLM to validate if the extracted answer matches the full response.
    Returns a validated answer and confidence score.
    """
    if task_type == 'unknown':
        prompt = validation_prompt.format(
            question=question,
            answer=answer,
            full_response=full_response,
        )
    else:
        prompt = validation_prompt.format(
            task_type=task_type,
            question=question,
            answer=answer,
            full_response=full_response,
            answer_spec=task_type_to_answer_type[task_type]
        )

    response = client.messages.create(
        model=judge,
        max_tokens=300,
        temperature=0,
        messages=[{
            "role": "user",
            "content": prompt
        }],
    )

    # Extract the three attributes from the response text
    extracted_answer = None
    correct = None
    explanation = None
    
    # Split the response into lines and look for the key patterns
    lines = response.content[0].text.split('\n')
    for line in lines:
        if line.startswith('Extracted Answer:'):
            extracted_answer = line.split(':')[1].strip()
            if extracted_answer.lower() == 'none':
                extracted_answer = None
        elif line.startswith('Correct:'):
            correct = line.split(':')[1].strip().lower() == 'true'
        elif line.startswith('Explanation (if \'None\'):'):
            explanation = line.split(':')[1].strip()
    
    # Create validation dictionary
    validation = {
        'correct': correct,
        'extracted_answer': extracted_answer,
        'explanation': explanation
    }
    return ValidationResult(**validation)
    
def get_extreme_point_descriptions(points, colors, shapes, coordinate_idx, get_extreme_fn=min):
    """Helper function for min/max x/y tasks
    Args:
        points: List of coordinate pairs
        colors: List of color strings
        shapes: List of shape strings
        coordinate_idx: 0 for x-coordinate, 1 for y-coordinate
        get_extreme_fn: Function to get extreme value (min or max)
    """
    values = [p[coordinate_idx] for p in points]
    extreme_val = get_extreme_fn(values)
    extreme_indices = [i for i, v in enumerate(values) if v == extreme_val]
    ans = [f"{colors[i]} {shapes[i]}" for i in extreme_indices]
    
    # Add alternatives
    new_ans = []
    for item in ans:
        new_ans.append(item)
        # Handle orange/yellow alternatives
        if "orange" in item:
            shape = item.split()[1]
            new_ans.append(f"yellow {shape}")
            if shape == "circle":
                new_ans.append(f"yellow dot")

        if "circle" in item:
            new_ans.append(item.replace("circle", "dot"))
        
        # Check for unique properties
        color, shape = item.split()

        # Check if this is the only object with this shape
        shape_count = sum(1 for s in shapes if s == shape)
        if shape_count == 1:
            new_ans.append(f"the {shape}")
            if shape == "circle":
                new_ans.append(f"the dot")

        color_count = sum(1 for c in colors if c == color)
        if color_count == 1:
            new_ans.append(f"{color}")
    
    return new_ans[0] if len(new_ans) == 1 else new_ans
    
def get_task_answer(task_row, task=None):
    if task is None:
        task_type = task_row['task_type']
    else:
        task_type = task
    
    if task_type == 'unknown':
        return task_row['answer']

    if task_type not in ['correlation', 'function', 'cluster', 'outlier']:
        points = task_row['grid_points'] if isinstance(task_row['grid_points'], list) else json.loads(task_row['grid_points'])
        x = [p[0] for p in points]
        y = [p[1] for p in points]
        colors = task_row['color'] if isinstance(task_row['color'], list) else json.loads(task_row['color'])
        shapes = task_row['dot_shape'] if isinstance(task_row['dot_shape'], list) else json.loads(task_row['dot_shape'])
        shapes = [SYMBOL_TO_SHAPE_MAP[s] for s in shapes]

    if "chart_type" in task_row:
        chart_type = task_row['chart_type']
    else:
        chart_type = None

    if task_type == 'count':
        return task_row['n_points']
    elif task_type == 'position':
        if chart_type is None:
            ans = json.loads(task_row['answer'])[0]
            return tuple(ans)
        else:
            ans = float(task_row['answer'].split('[')[-1].split(']')[0].split('(')[-1].split(')')[0])
            return ans
    elif task_type == 'distance':
        possible_answers = json.loads(task_row['answer'])
        possible_answers = list(dict.fromkeys(possible_answers))  # Remove duplicates while preserving order
        return possible_answers[0] if len(possible_answers) == 1 else possible_answers
    elif task_type == 'min_x':
        if chart_type == 'bar':
            min_idx = np.argmin(x)
            return colors[min_idx]
        elif chart_type == 'line':
            return json.loads(task_row['answer'])[0]
        else:
            return get_extreme_point_descriptions(points, colors, shapes, coordinate_idx=0, get_extreme_fn=min)
    elif task_type == 'min_y':
        if chart_type == 'bar':
            min_idx = np.argmin(y)
            return colors[min_idx]
        elif chart_type == 'line':
            return json.loads(task_row['answer'])[0]
        else:
            return get_extreme_point_descriptions(points, colors, shapes, coordinate_idx=1, get_extreme_fn=min)
    elif task_type == 'max_x':
        if chart_type == 'bar':
            max_idx = np.argmax(x)
            return colors[max_idx]
        elif chart_type == 'line':
            return json.loads(task_row['answer'])[0]
        else:
            return get_extreme_point_descriptions(points, colors, shapes, coordinate_idx=0, get_extreme_fn=max)
    elif task_type == 'max_y':
        if chart_type == 'bar':
            max_idx = np.argmax(y)
            return colors[max_idx]
        elif chart_type == 'line':
            return json.loads(task_row['answer'])[0]
        else:
            return get_extreme_point_descriptions(points, colors, shapes, coordinate_idx=1, get_extreme_fn=max)
    elif task_type == 'mean':
        if chart_type is None:
            answer = task_row['answer'].strip('[]').split('), (')
            answer = [tuple(map(int, pair.strip('()').split(','))) for pair in answer]
            answer = list(dict.fromkeys(answer))  # Remove duplicates while preserving order
        else:
            answer = task_row['answer'].strip('[]').split(',')
            answer = [float(a) for a in answer]
        return answer
    elif task_type in ['correlation', 'function', 'cluster']:
        answer = task_row['answer'].strip("[]'")
        return answer
    elif task_type == 'outlier':
        answer = task_row['answer'].strip("[]()").split('(')
        answer = [int(answer[1].split(',')[0].strip(')')), int(answer[-1])]
        return answer
    else:
        raise ValueError(f"Unknown task type: {task_type}")

def load_results(results_dir, results_subdir=None):
    result_files = glob.glob(f"{results_dir}/*.json")
    if results_subdir not in ["charxiv"]:
        results = []
        for f in result_files:
            try:
                result = json.load(open(f, encoding='utf-8'))
                result['validation_prompt'] = DEFAULT_VALIDATION_PROMPT
                results.append(result)
            except json.decoder.JSONDecodeError:
                continue
        results = pd.DataFrame(results)
    else:
        results = {}
        for f in result_files:
            try:
                result = json.load(open(f, encoding='utf-8'))
                result['response'] = result['full_response']
                result['validation_prompt'] = CHARXIV_VALIDATION_PROMPT
                result['task_type'] = 'unknown'
                results[result['id']] = result
            except json.decoder.JSONDecodeError:
                continue

        with open("/data/alexart/fugu-iclr/CharXiv/data/descriptive_val.json") as f:
            data = json.load(f)

        from CharXiv.src.descriptive_utils import preprocess_descriptive_grading_queries, build_descriptive_grading_queries

        groups = preprocess_descriptive_grading_queries(data, results)
        queries = build_descriptive_grading_queries(groups)

        for query in queries:
            qids = query['resp_keys']
            prompt = query['grading_query'].split("Overarching Question: ")[-1].split("\n")[0]
            for i, qid in enumerate(qids):
                ground_truth = query['grading_query'].split(f"Ground Truth {i + 1}: ")[-1].split("\n")[0]

                results[qid]['answer'] = ground_truth

        results = pd.DataFrame(results.values())
    
    return results
    
def analyze_answers_with_llm(model_id, perform_cot_analysis=False, results_dir="results", results_subdir=None):
    model_id = model_id.split('/')[-1]
    if results_subdir is not None:
        results_dir = f"{results_dir}/{model_id}/{results_subdir}"
    else:
        results_dir = f"{results_dir}/{model_id}"

    results = load_results(results_dir, results_subdir)
    
    # Create behavioral_analysis directory if it doesn't exist
    analysis_dir = f"{results_dir}/{"cot_analysis" if perform_cot_analysis else "behavioral_analysis"}"
    os.makedirs(analysis_dir, exist_ok=True)

    if perform_cot_analysis:
        cot_results = results[results['generation_type'] == 'cot']
        if len(cot_results) == 0:
            results = results[results['generation_type'] == 'none']
        else:
            results = cot_results

    validated_results = pd.DataFrame()

    total = len(results)
    progress_bar = tqdm.tqdm(total=total, desc="Processing results")
    skip_validation = False

    for task_type in results['task_type'].unique():
        task_results = results[results['task_type'] == task_type]

        for _, row in task_results.iterrows():
            # Save individual validated result
            if 'generation_type' in row:
                if row['generation_type'] in ['immediate', 'ground_truth_listing', 'model_listing', 'implicit', 'visual_strategy', 'text_only']:
                    cot_str = f"_{row['generation_type']}"
                else:
                    cot_str = "_cot"
            else:
                cot_str = "_cot"
            result_file = f"{analysis_dir}/{row['id']}_{row['task_type']}{cot_str}.json"

            if os.path.exists(result_file):
                if perform_cot_analysis:
                    with open(result_file, 'r', encoding='utf-8') as f:
                        existing_result = json.load(f)
                        if all(f'cot_analysis_{key}' in existing_result for key in ['lists_points', 'explanation']):
                            progress_bar.update(1)
                            continue
                        else:
                            skip_validation = True
                else:
                    progress_bar.update(1)
                    continue

            if not skip_validation:
                answer = get_task_answer(row)
                
                validation = validate_extraction_with_llm(
                    row['task_type'], 
                    row['prompt'],
                    answer,
                    row['full_response'],
                    validation_prompt=row['validation_prompt'],
                )

                for key, value in validation.items():
                    row[f'validation_{key}'] = value

            if perform_cot_analysis:
                if results_subdir != "charxiv":
                    color = row['color']
                    dot_shape = row['dot_shape']
                    points_val = row['grid_points']

                    if isinstance(color, str) and "[" in color:
                        color = json.loads(color)
                    else:
                        color = color
                    if isinstance(dot_shape, str) and "[" in dot_shape:
                        dot_shape = json.loads(dot_shape)
                    else:
                        dot_shape = dot_shape
                    if isinstance(points_val, str) and "[" in points_val:
                        points = json.loads(points_val)
                    else:
                        points = points_val

                    #ground_truth_objects = [f"{c} {SYMBOL_TO_SHAPE_MAP[s]}" for c, s in zip(color, dot_shape)]
                    points = [(round(p[0]), round(p[1])) for p in points]
                    if 'chart_type' in row:
                        chart_type = row['chart_type']
                    else:
                        chart_type = 'scatter'
                else:
                    points = "unknown"
                    chart_type = "unknown"

                cot_analysis = analyze_cot_with_llm(
                    row['full_response'],
                    points,
                    chart_type,
                )

                for key, value in cot_analysis.items():
                    row[f'cot_analysis_{key}'] = value
            
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(row.to_dict(), f, indent=2)
            
            # Still append to validated_results for the final CSV
            validated_results = pd.concat([validated_results, pd.DataFrame([row])], ignore_index=True)
            progress_bar.update(1)
            skip_validation = False
    
    progress_bar.close()
        
    validated_results.to_csv(f"{analysis_dir}/validated_results.csv", index=False)
    return validated_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, required=True, 
                        help="Model ID to analyze",
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
                                "chartqa/Llama-3.2-11B-Vision-Instruct",
                                "google/gemma-3-12b-it",
                                "gemini-2.5-pro",
                                "gemini-2.5-flash"])
    parser.add_argument("--perform_cot_analysis", action="store_true",
                        help="Whether to perform COT analysis")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--results_subdir", type=str, default=None)
    args = parser.parse_args()
    _ = analyze_answers_with_llm(args.model_id, args.perform_cot_analysis, args.results_dir, args.results_subdir)