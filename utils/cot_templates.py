import pandas as pd
import glob
import json
import os

import torch
import numpy as np
from PIL import Image

from models.models import get_model_handler
from utils.generate_prompts import SYMBOL_TO_SHAPE_MAP


COT_TEMPLATES = {
    "ground_truth_listing": """
STEP 1: **Identify the (x, y) coordinates of all points in the image.**
{point_list}

STEP 2: **State {target}**
BE CONCISE; ONLY RESPOND WITH THE ANSWER. {target_capitalized} = """,

    "model_listing": """
STEP 1: Let's identify each point in the image one by one.
{interactive_point_list}

STEP 2: **State {target}**
BE CONCISE; ONLY RESPOND WITH THE ANSWER. {target_capitalized} = """,

    "implicit": """
STEP 1: **Identify the (x, y) coordinates of the data points.**
- Look at each point in the image and note their positions.

STEP 2: **State {target}**
BE CONCISE; ONLY RESPOND WITH THE ANSWER. {target_capitalized} = """,

    "immediate": """
STEP 1: **State {target}**
BE CONCISE; ONLY RESPOND WITH THE ANSWER. {target_capitalized} = """,
}

VISUAL_COT_TEMPLATES = {
    "count": """
STEP 1: **Use the image to count the data points.**
- Scan the image systematically from left to right and top to bottom
- Keep track of each point as you come across

STEP 2: **State {target}**
BE CONCISE; ONLY RESPOND WITH THE ANSWER. {target_capitalized} = """,
    "position": """
STEP 1: **Use the image to determine the (x, y) coordinates of the data point.**
- Scan the image and identify any {color} points
- From the {color} points, identify the point with a {shape} shape
- Read off the x and y coordinates for this point

STEP 2: **State {target}**
BE CONCISE; ONLY RESPOND WITH THE ANSWER. {target_capitalized} = """,
    "distance": """
STEP 1: **Use the image to determine the distance between two points.**
- First, locate the two points we need to measure between
- Visualize a straight line connecting these points
- Count the grid units in both the horizontal (x) and vertical (y) directions
- Use these measurements to calculate the total distance (like a right triangle)

STEP 2: **State {target}**
BE CONCISE; ONLY RESPOND WITH THE ANSWER. {target_capitalized} = """,
    "min_x": """
STEP 1: **Use the image to identify the point with the smallest x-value.**
- Scan the image from left to right
- Identify the first point you come across, i.e. the leftmost point
- This will have the smallest x-value

STEP 2: **State {target}**
BE CONCISE; ONLY RESPOND WITH THE ANSWER. {target_capitalized} = """,
    "max_x": """
STEP 1: **Use the image to identify the point with the largest x-value.**
- Scan the image from right to left
- Identify the first point you come across, i.e. the rightmost point
- This will have the largest x-value
    
STEP 2: **State {target}**
BE CONCISE; ONLY RESPOND WITH THE ANSWER. {target_capitalized} = """,
    "min_y": """
STEP 1: **Use the image to identify the point with the smallest y-value.**
- Scan the image from bottom to top
- Identify the first point you come across, i.e. the lowest point
- This will have the smallest y-value
    
STEP 2: **State {target}**
BE CONCISE; ONLY RESPOND WITH THE ANSWER. {target_capitalized} = """,
    "max_y": """
STEP 1: **Use the image to identify the point with the largest y-value.**
- Scan the image from top to bottom
- Identify the first point you come across, i.e. the highest point
- This will have the largest y-value
    
STEP 2: **State {target}**
BE CONCISE; ONLY RESPOND WITH THE ANSWER. {target_capitalized} = """,
    "mean": """
STEP 1: **Use the image to estimate the centroid of the points.**
- Look at the point-cloud in the image
- Estimate where the center of the point-cloud might be
- The grid point closest to this central position will represent the centroid

STEP 2: **State {target}**
BE CONCISE; ONLY RESPOND WITH THE ANSWER. {target_capitalized} = """, 
}

def get_forced_answer_target(task_type, prompt, root_dir=""):
    if task_type == "count":
        return "the number of data points in the plot"
    elif task_type == "position":
        if "bar_chart" in root_dir or "line_chart" in root_dir:
            return prompt.split("What is ")[1].replace("?", "")
        return prompt.split("What is ")[1].replace("? Please format your response", ", given").replace(".", "")
    elif task_type == "distance":
        return prompt.split("What is ")[1].replace("?", "")
    elif "min" in task_type or "max" in task_type:
        if "bar_chart" in root_dir:
            stub = prompt.split("Which bar has the ")[1].split("?")[0]
            return f"the color of the bar with the {stub}"
        elif "line_chart" in root_dir:
            stub = prompt.split("What is ")[1].split("?")[0]
            return f"the y-value of the data point with {stub}"
        stub = prompt.split("Which data point has the ")[1].split("?")[0]
        return f"the color and shape of the data point with the {stub}"
    elif task_type == "mean":
        if "bar_chart" in root_dir:
            return "the average value of the bars, rounded to the nearest whole number"
        elif "line_chart" in root_dir:
            return "the average y-value of the data points, rounded to the nearest whole number"
        return "the (x, y) coordinate that is closest to the centroid, rounded to the nearest whole number"
    elif task_type == "correlation":
        return "whether the correlation coefficient is higher or lower than 0.5"
    elif task_type == "function":
        return "which function best describes the pattern in the data"
    elif task_type == "cluster":
        return "which cluster the query point belongs to"
    elif task_type == "outlier":
        return "the (x, y) coordinate of the outlier point rounded to the nearest integer"
    else:
        raise ValueError(f"Unknown task type: {task_type}")
    
def get_ground_truth_point_list(row, root_dir="datasets/bar_chart/bar_charts"):
    metadata = np.load(f"{root_dir}/metadata/{row['id']}.npy", allow_pickle=True).item()

    if "bar_chart" in root_dir:
        colors = metadata['color']
        labels = metadata['x_tick_labels']
        if labels is None:
            labels = metadata['y_tick_labels']
        points = [a[1] for a in metadata['grid_points']]
        return "\n".join([f"- {color} ({label}): value={value}" for color, label, value in zip(colors, labels, points)])
    elif "line_chart" in root_dir:
        points = metadata['grid_points']
        return "\n".join([f"- x={coords[0]}: y={coords[1]}" for coords in points])
    else:
        colors = metadata['color']
        shapes = [SYMBOL_TO_SHAPE_MAP[s] for s in metadata['dot_shape']]
        points = metadata['points']
        return "\n".join([f"- {color} {shape}: x={coords[0]}, y={coords[1]}" for color, shape, coords in zip(colors, shapes, points)])
    
def get_interactive_point_list(model, row, root_dir="datasets/bar_chart/bar_charts"):
    model_name = model.model_id.split('/')[-1]
    dataset_name = root_dir.split('/')[-1]

    # Check if model prompted point list already exists
    if os.path.exists(f"results/interactive_point_lists/{model_name}_{dataset_name}_{row['id']}.txt"):
        return open(f"results/interactive_point_lists/{model_name}_{dataset_name}_{row['id']}.txt").read()
    
    # If not, prompt model to generate point list
    metadata = np.load(f"{root_dir}/metadata/{row['id']}.npy", allow_pickle=True).item()

    if "bar_chart" in root_dir:
        colors = metadata['color']
        labels = metadata['x_tick_labels']
        if labels is None:
            labels = metadata['y_tick_labels']
        
        position_prompt = "What is the value of the {color} bar (labeled {label})? BE CONCISE; ONLY RESPOND WITH THE NUMBER. The value of the {color} bar (labeled {label}) = "
        responses = {}

        for color, label in zip(colors, labels):
            k = f"{color.capitalize()} bar ({label})"
            responses[k] = 0
            prompt = position_prompt.format(color=color, label=label)
            image_path = f"{root_dir}/stimuli/{row['id']}.png"

            inputs = model.process_input(image=Image.open(image_path), prompt=prompt)
            answer = model.generate(**inputs)
            responses[k] = float(answer)

        point_list = "\n".join([f"- {color_bar}: value={value}" for color_bar, value in responses.items()])

    elif "line_chart" in root_dir:
        x_points = [a[0] for a in metadata['grid_points']]

        position_prompt = "What is the y-value of the data point at x = {x}? BE CONCISE; ONLY RESPOND WITH THE NUMBER. The y-value of the data point at x = {x} = "
        responses = {}

        for x in x_points:
            k = f"x = {x}"
            responses[k] = 0
            prompt = position_prompt.format(x=x)
            image_path = f"{root_dir}/stimuli/{row['id']}.png"

            inputs = model.process_input(image=Image.open(image_path), prompt=prompt)
            answer = model.generate(**inputs)
            responses[k] = float(answer)

        point_list = "\n".join([f"- x = {x}: y = {y}" for x, y in responses.items()])

    else:
        colors = metadata['color']
        shapes = [SYMBOL_TO_SHAPE_MAP[s] for s in metadata['dot_shape']]

        position_prompt = "What is the {axis}-value of the {color} {shape}? BE CONCISE; ONLY RESPOND WITH THE NUMBER. The {axis}-value of the {color} {shape} = "
        responses = {}

        for color, shape in zip(colors, shapes):
            k = f"{color.capitalize()} {shape}"
            responses[k] = [0, 0]
            for a, axis in enumerate(['x', 'y']):
                prompt = position_prompt.format(axis=axis, color=color, shape=shape)
                image_path = f"{root_dir}/stimuli/{row['id']}.png"

                inputs = model.process_input(image=Image.open(image_path), prompt=prompt)
                answer = model.generate(**inputs)
                responses[k][a] = float(answer)

        point_list = "\n".join([f"- {color_shape}: x={coords[0]}, y={coords[1]}" for color_shape, coords in responses.items()])

    os.makedirs("results/interactive_point_lists", exist_ok=True)
    with open(f"results/interactive_point_lists/{model_name}_{dataset_name}_{row['id']}.txt", "w") as f:
        f.write(point_list)

    return point_list

def get_cot_template(model, row, cot_type="ground_truth_listing", root_dir="datasets/bar_chart/bar_charts"):
    if cot_type is None:
        return None  # If no CoT type is specified, then allow the model to freely generate its own answer
    elif cot_type == "text_only":
        pass
    elif cot_type not in COT_TEMPLATES and cot_type != "visual_strategy":
        raise ValueError(f"Invalid COT type: {cot_type}")
    
    target = get_forced_answer_target(row['task_type'], row['prompt'], root_dir)
    
    if cot_type == "ground_truth_listing" or cot_type == "text_only":
        point_list = get_ground_truth_point_list(row, root_dir)
        prompt = COT_TEMPLATES["ground_truth_listing"].format(point_list=point_list, target=target, target_capitalized=target.capitalize())
    elif cot_type == "model_listing":
        point_list = get_interactive_point_list(model, row, root_dir)
        prompt = COT_TEMPLATES[cot_type].format(interactive_point_list=point_list, target=target, target_capitalized=target.capitalize())
    elif cot_type == "implicit":
        prompt = COT_TEMPLATES[cot_type].format(target=target, target_capitalized=target.capitalize())
    elif cot_type == "visual_strategy":
        obj = target.split("coordinate of the ")[-1].split(",")[0].split(" ")
        prompt = VISUAL_COT_TEMPLATES[row['task_type']].format(
            target=target,
            target_capitalized=target.capitalize(),
            color=obj[0],
            shape=obj[1]
        )
    elif cot_type == "immediate":
        prompt = COT_TEMPLATES[cot_type].format(target=target, target_capitalized=target.capitalize())
    
    return prompt