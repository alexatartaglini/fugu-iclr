import glob
import math
import pandas as pd
import numpy as np
import json
import yaml
from typing import List, Dict, Tuple
from pathlib import Path


PROMPT_PREFIX = """
The image shows either a bar chart or a line chart displaying quantitative values along the x- and y-axes.
Bars always use unique colors chosen from an extended palette, while line charts connect markers drawn in a single shared color (circles, triangles, squares, or stars) across all points.
Each bar height or line-marker position matches its exact x- and y-values.
Please answer the following question based on the information conveyed by this chart.
"""

PROMPT_PREFIX_CONCISE = """
The image shows either a bar chart or a line chart displaying quantitative values along the x- and y-axes.
Bars always use unique colors chosen from an extended palette, while line charts connect markers drawn in a single shared color (circles, triangles, squares, or stars) across all points.
Each bar height or line-marker position matches its exact x- and y-values.
Please answer the following question based on the information conveyed by this chart.
Please provide an exact numerical response and no additional explanation.
"""

PROMPT_PREFIX_BAR = """
The image shows a bar chart in which one axis is categorical (labeled with letters) and the other axis is quantitative.
Each bar corresponds to a single category and its height encodes the exact numerical value associated with that category.
Bars appear in eight distinct colors (red, green, blue, orange, cyan, magenta, brown, or purple), with each bar using exactly one of these colors.
Please answer the following question based on the information conveyed by this bar chart.
"""

PROMPT_PREFIX_BAR_CONCISE = """
The image shows a bar chart in which one axis is categorical (labeled with letters) and the other axis is quantitative.
Each bar corresponds to a single category and its height encodes the exact numerical value associated with that category.
Bars appear in eight distinct colors (red, green, blue, orange, cyan, magenta, brown, or purple), with each bar using exactly one of these colors.
Please answer the following question based on the information conveyed by this bar chart.
Please provide an exact numerical response and no additional explanation.
"""

PROMPT_PREFIX_LINE = """
The image shows a line chart displaying the relationship between two quantitative variables, labeled 'x' (horizontal axis) and 'y' (vertical axis).
Each observation is shown as a marker (circle, triangle, square, or star) placed according to its exact x- and y-values, and consecutive markers are connected by a line.
All markers and the connecting line appear in a single shared color.
Please answer the following question based on the information conveyed by this line chart.
"""

PROMPT_PREFIX_LINE_CONCISE = """
The image shows a line chart displaying the relationship between two quantitative variables, labeled 'x' (horizontal axis) and 'y' (vertical axis).
Each observation is shown as a marker (circle, triangle, square, or star) placed according to its exact x- and y-values, and consecutive markers are connected by a line.
All markers and the connecting line appear in a single shared color.
Please answer the following question based on the information conveyed by this line chart.
Please provide an exact numerical response and no additional explanation.
"""


SYMBOL_TO_SHAPE_MAP = {
    "o": "circle",
    "s": "square",
    "^": "triangle",
    "*": "star"
}


def describe_datapoint(index: int, metadata: Dict) -> str:
    chart_type = metadata.get('chart_type', 'scatter')
    colors = metadata['color']
    shapes = metadata['dot_shape']
    shape_name = SYMBOL_TO_SHAPE_MAP.get(shapes[index], shapes[index])

    if chart_type == 'bar':
        return f"{colors[index]} bar"
    elif chart_type == 'line':
        return f"{colors[index]} {shape_name} marker"
    return f"{colors[index]} {shape_name}"

QUERIES = {
    "count": {
        "prompt": "How many data points or bars are there in this chart?",
        "answer_template": "The number of visual elements in the chart is "
    },
    "position": {
        "prompt": "What is the (x, y) coordinate of the {target}? Please format your response as an ordered pair of values enclosed within parentheses.",
        "answer_template": "The (x, y) coordinate of the {target} is ("
    },
    "distance": {
        "prompt": "What is the distance between the {target1} and the {target2}, rounded to the nearest whole number?",
        "answer_template": "The distance between the {target1} and the {target2} rounded to the nearest whole number is "
    },
    "min": {
        "prompt": "Which data point has the smallest {axis}-value? Please identify this data point.",
        "answer_template": "The data point with the smallest {axis}-value is the "
    },
    "max": {
        "prompt": "Which data point has the largest {axis}-value? Please identify this data point.",
        "answer_template": "The data point with the largest {axis}-value is the "
    },
    "mean": {
        "prompt": "What is the (x, y) coordinate that is closest to the centroid, or arithmetic mean of the positions of all data points? Please round both the x-value and y-value to the nearest whole number.",
        "answer_template": "The (x, y) coordinate that is closest to the centroid (arithmetic mean of all data points) is ("
    },
    "correlation": {
        "prompt": "Is the correlation coefficient {option_1} or {option_2} than {threshold}?",  # higher or lower
        "answer_template": "The correlation coefficient is "
    },
    "cluster": {
        "prompt": "Which cluster does the {query_color} {query_shape} most likely belong to: the {color1} cluster or the {color2} cluster?",
        "answer_template": "The {query_color} {query_shape} belongs to the "
    },
    "function": {
        "prompt": "Which function best describes the pattern in the data: {option_1} or {option_2}?",  # linear, quadratic, exponential, or logarithmic
        "answer_template": "The data is best described by a"
    },
    "outlier": {
        "prompt": "There is an outlier in the {stimulus_type}. What is the (x, y) coordinate of the outlier point rounded to the nearest integer?",
        "answer_template": "The (x, y) coordinate of the outlier point rounded to the nearest integer is ("
    }
}

BAR_QUERIES = {
    "count": {
        "prompt": "How many bars are shown in this chart?",
        "answer_template": "The number of bars in the chart is "
    },
    "value": {
        "prompt": "What is the value (height) of the {color} bar labeled '{category}'?",
        "answer_template": "The value of the {color} bar for category '{category}' is "
    },
    "min": {
        "prompt": "Which bar has the smallest value? Identify it by its color.",
        "answer_template": "The bar with the smallest value is the "
    },
    "max": {
        "prompt": "Which bar has the largest value? Identify it by its color.",
        "answer_template": "The bar with the largest value is the "
    },
    "mean": {
        "prompt": "What is the average (mean) value across all bars? Round to the nearest whole number.",
        "answer_template": "The average value across all bars, rounded to the nearest whole number, is "
    },
    "distance": {
        "prompt": "What is the difference in value between the {color1} bar (category '{category1}') and the {color2} bar (category '{category2}'), rounded to the nearest whole number?",
        "answer_template": "The difference in value between the {color1} '{category1}' bar and the {color2} '{category2}' bar is "
    }
}

LINE_QUERIES = {
    "count": {
        "prompt": "How many markers (data points) are plotted in this line chart?",
        "answer_template": "The number of data points in the line chart is "
    },
    "position": {
        "prompt": "What is the y-value of the data point at x = {x}?",
        "answer_template": "The y-value of the data point at x = {x} is "
    },
    "distance": {
        "prompt": "What is the difference in y-values between the data point at x = {x1} and the data point at x = {x2}?",
        "answer_template": "The difference in y-values between the data point at x = {x1} and the data point at x = {x2} is "
    },
    "min": {
        "prompt": "Which data point has the smallest y-value? Identify it by its x-value.",
        "answer_template": "The data point with the smallest y-value is the "
    },
    "max": {
        "prompt": "Which data point has the largest y-value? Identify it by its x-value.",
        "answer_template": "The data point with the largest y-value is the "
    },
    "mean": {
        "prompt": "What is the average y-value of all data points? Round to the nearest whole number.",
        "answer_template": "The average y-value of all data points, rounded to the nearest whole number, is "
    }
}


# Define main query types (for non-ensemble stimuli) and ensemble query types
MAIN_QUERY_TYPES = ['count', 'position', 'distance', 'min_x', 'min_y', 'max_x', 'max_y', 'mean']
ENSEMBLE_QUERY_TYPES = ['correlation', 'function', 'outlier', 'cluster']


def to_relative(coord, metadata):
    return (coord * metadata['tick_scale']) / metadata['axis_range'][1]


def rounded_relative(coord, metadata):
    if 'tick_scale' not in metadata or 'n_ticks' not in metadata:
        return coord
    
    round_neighborhood = max(1, metadata['tick_scale'] // (metadata['n_ticks']*2))
    
    # Round coord up or down within round_neighborhood
    if isinstance(coord, (int, float)):
        # For single value
        rounded = round(coord / round_neighborhood) * round_neighborhood
        return rounded
    else:
        # For array-like coordinates
        rounded = np.round(coord / round_neighborhood) * round_neighborhood
        return rounded


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_stimuli_metadata(root_dir: str) -> pd.DataFrame:
    """Load all stimuli metadata from the specified root directory."""

    root_path = Path(root_dir)
    if not root_path.exists():
        raise FileNotFoundError(f"Root directory does not exist: {root_dir}")

    metadata_dirs = [root_path / "metadata"]
    metadata_dirs.extend(
        p for p in root_path.glob("**/metadata") if p != metadata_dirs[0]
    )

    metadata = []
    for metadata_dir in metadata_dirs:
        if not metadata_dir.exists():
            continue

        stimuli_files = glob.glob(str(metadata_dir / "*.npy"))
        for file in stimuli_files:
            try:
                data = np.load(file, allow_pickle=True).item()
                data['metadata_path'] = file
                metadata.append(data)
            except Exception as e:
                print(f"Error loading {file}: {e}")
                continue

    return pd.DataFrame(metadata)


def is_ensemble_stimulus(metadata: Dict) -> bool:
    """Check if a stimulus is an ensemble stimulus based on generator_metadata"""
    if 'generator_metadata' not in metadata:
        return False
    
    gen_metadata = metadata['generator_metadata']
    if 'stimulus_type' not in gen_metadata:
        return False
    
    return gen_metadata['stimulus_type'] in ENSEMBLE_QUERY_TYPES


def separate_stimuli(stimuli_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Separate stimuli into main (non-ensemble) and ensemble stimuli"""
    ensemble_mask = stimuli_df.apply(lambda row: is_ensemble_stimulus(row.to_dict()), axis=1)
    
    main_stimuli = stimuli_df[~ensemble_mask].copy()
    ensemble_stimuli = stimuli_df[ensemble_mask].copy()
    
    return main_stimuli, ensemble_stimuli


def get_applicable_task_types(n_points: int) -> List[str]:
    """Get list of applicable task types for given number of points"""
    if n_points == 1:
        return ['count', 'position']
    elif n_points == 2:
        return ['count', 'position', 'distance']
    else:
        return ['count', 'position', 'distance', 'min_x', 'min_y', 'max_x', 'max_y', 'mean']


def generate_all_prompts(main_stimuli: pd.DataFrame, ensemble_stimuli: pd.DataFrame) -> pd.DataFrame:
    """Generate all possible prompts for all stimuli"""
    all_prompts = []
    
    # Generate all main prompts
    for _, stim in main_stimuli.iterrows():
        n_points = stim['n_points']
        applicable_tasks = get_applicable_task_types(n_points)
        
        for task_type in applicable_tasks:
            all_prompts.append({
                'id': stim['id'],
                'metadata_path': stim['metadata_path'],
                'task_type': task_type,
                'n_points': n_points,
                'prompt': None,
                'answer': None,
                'answer_template': None,
            })
    
    # Generate all ensemble prompts
    for _, stim in ensemble_stimuli.iterrows():
        gen_metadata = stim['generator_metadata']
        stimulus_type = gen_metadata['stimulus_type']
        
        if stimulus_type in ENSEMBLE_QUERY_TYPES:
            all_prompts.append({
                'id': stim['id'],
                'metadata_path': stim['metadata_path'],
                'task_type': stimulus_type,
                'n_points': stim.get('n_points', None),
                'prompt': None,
                'answer': None,
                'answer_template': None,
            })
    
    return pd.DataFrame(all_prompts)


def generate_config_based_main_prompts(main_stimuli: pd.DataFrame, prompt_config: Dict) -> pd.DataFrame:
    """
    Generate main prompts based on config specification with balanced sampling
    """
    targets = prompt_config['targets']
    balance_across = prompt_config.get('balance_across', ['n_points', 'task_type'])
    allow_replacement = prompt_config.get('allow_replacement', False)
    
    # Handle extremum task splitting
    expanded_targets = {}
    for task, count in targets.items():
        if task == 'min':
            expanded_targets['min_x'] = count // 2
            expanded_targets['min_y'] = count - (count // 2)  # Handle odd numbers
        elif task == 'max':
            expanded_targets['max_x'] = count // 2
            expanded_targets['max_y'] = count - (count // 2)  # Handle odd numbers
        else:
            expanded_targets[task] = count
    
    selected_prompts = []
    
    # Get unique n_points values for balancing
    unique_n_points = sorted(main_stimuli['n_points'].unique())
    
    # Generate prompts for each task type
    for task_type, target_count in expanded_targets.items():
        print(f"Generating {target_count} prompts for task: {task_type}")
        
        # Filter stimuli that can handle this task type
        applicable_stimuli = []
        for n_points in unique_n_points:
            if task_type in get_applicable_task_types(n_points):
                n_point_stimuli = main_stimuli[main_stimuli['n_points'] == n_points]
                applicable_stimuli.append((n_points, n_point_stimuli))
        
        if not applicable_stimuli:
            print(f"Warning: No applicable stimuli for task {task_type}")
            continue
        
        # Balance across n_points if specified
        if 'n_points' in balance_across and len(applicable_stimuli) > 1:
            prompts_per_n_points = target_count // len(applicable_stimuli)
            extra_prompts = target_count % len(applicable_stimuli)
            
            for i, (n_points, n_point_stimuli) in enumerate(applicable_stimuli):
                n_prompts = prompts_per_n_points + (1 if i < extra_prompts else 0)
                n_prompts = min(n_prompts, len(n_point_stimuli))
                
                # Sample stimuli
                if allow_replacement or n_prompts <= len(n_point_stimuli):
                    sampled = n_point_stimuli.sample(n=n_prompts, replace=allow_replacement)
                else:
                    sampled = n_point_stimuli  # Use all available
                
                for _, stim in sampled.iterrows():
                    selected_prompts.append({
                        'id': stim['id'],
                        'metadata_path': stim['metadata_path'],
                        'task_type': task_type,
                        'n_points': n_points,
                        'prompt': None,
                        'answer': None,
                        'answer_template': None,
                    })
        else:
            # No n_points balancing, sample from all applicable stimuli
            all_applicable = pd.concat([stimuli for _, stimuli in applicable_stimuli], ignore_index=True)
            n_prompts = min(target_count, len(all_applicable))
            
            if allow_replacement or n_prompts <= len(all_applicable):
                sampled = all_applicable.sample(n=n_prompts, replace=allow_replacement)
            else:
                sampled = all_applicable
            
            for _, stim in sampled.iterrows():
                selected_prompts.append({
                    'id': stim['id'],
                    'metadata_path': stim['metadata_path'],
                    'task_type': task_type,
                    'n_points': stim['n_points'],
                    'prompt': None,
                    'answer': None,
                    'answer_template': None,
                })
    
    return pd.DataFrame(selected_prompts)


def generate_config_based_ensemble_prompts(ensemble_stimuli: pd.DataFrame, prompt_config: Dict) -> pd.DataFrame:
    """
    Generate ensemble prompts based on config specification
    """
    targets = prompt_config['targets']
    balance_across = prompt_config.get('balance_across', ['n_points', 'task_type'])
    sampling_strategy = prompt_config.get('sampling_strategy', 'one_per_stimulus')
    
    selected_prompts = []
    
    if sampling_strategy == 'one_per_stimulus':
        # Generate exactly one prompt per stimulus based on its type
        for _, stim in ensemble_stimuli.iterrows():
            gen_metadata = stim['generator_metadata']
            stimulus_type = gen_metadata['stimulus_type']
            
            if stimulus_type in ENSEMBLE_QUERY_TYPES:
                selected_prompts.append({
                    'id': stim['id'],
                    'metadata_path': stim['metadata_path'],
                    'task_type': stimulus_type,
                    'n_points': stim.get('n_points', None),
                    'prompt': None,
                    'answer': None,
                    'answer_template': None,
                })
    else:
        # Use target-based sampling with balancing
        for task_type, target_count in targets.items():
            if task_type not in ENSEMBLE_QUERY_TYPES:
                continue
                
            # Filter stimuli of this type
            task_stimuli = ensemble_stimuli[
                ensemble_stimuli['generator_metadata'].apply(
                    lambda x: x.get('stimulus_type') == task_type
                )
            ]
            
            if len(task_stimuli) == 0:
                print(f"Warning: No stimuli found for ensemble task {task_type}")
                continue
            
            # Balance across n_points if specified
            if 'n_points' in balance_across and 'n_points' in task_stimuli.columns:
                unique_n_points = sorted(task_stimuli['n_points'].unique())
                if len(unique_n_points) > 1:
                    prompts_per_n_points = target_count // len(unique_n_points)
                    extra_prompts = target_count % len(unique_n_points)
                    
                    for i, n_points in enumerate(unique_n_points):
                        n_prompts = prompts_per_n_points + (1 if i < extra_prompts else 0)
                        n_point_stimuli = task_stimuli[task_stimuli['n_points'] == n_points]
                        n_prompts = min(n_prompts, len(n_point_stimuli))
                        
                        sampled = n_point_stimuli.sample(n=n_prompts, replace=False)
                        for _, stim in sampled.iterrows():
                            selected_prompts.append({
                                'id': stim['id'],
                                'metadata_path': stim['metadata_path'],
                                'task_type': task_type,
                                'n_points': n_points,
                                'prompt': None,
                                'answer': None,
                                'answer_template': None,
                            })
                else:
                    # Only one n_points value, sample directly
                    n_prompts = min(target_count, len(task_stimuli))
                    sampled = task_stimuli.sample(n=n_prompts, replace=False)
                    for _, stim in sampled.iterrows():
                        selected_prompts.append({
                            'id': stim['id'],
                            'metadata_path': stim['metadata_path'],
                            'task_type': task_type,
                            'n_points': stim.get('n_points', None),
                            'prompt': None,
                            'answer': None,
                            'answer_template': None,
                        })
            else:
                # No n_points balancing
                n_prompts = min(target_count, len(task_stimuli))
                sampled = task_stimuli.sample(n=n_prompts, replace=False)
                for _, stim in sampled.iterrows():
                    selected_prompts.append({
                        'id': stim['id'],
                        'metadata_path': stim['metadata_path'],
                        'task_type': task_type,
                        'n_points': stim.get('n_points', None),
                        'prompt': None,
                        'answer': None,
                        'answer_template': None,
                    })
    
    return pd.DataFrame(selected_prompts)


def generate_balanced_main_prompts(main_stimuli: pd.DataFrame, target_total: int = 6000) -> pd.DataFrame:
    """
    Generate balanced prompts for main (non-ensemble) stimuli
    - All n==1 stimuli get position prompts
    - Each stimulus appears at least once
    - Remaining prompts balanced across query types and n_points
    """
    selected_prompts = []
    
    # Step 1: Position prompts for ALL n==1 stimuli
    n1_stimuli = main_stimuli[main_stimuli['n_points'] == 1]
    for _, stim in n1_stimuli.iterrows():
        selected_prompts.append({
            'id': stim['id'],
            'metadata_path': stim['metadata_path'],
            'task_type': 'position',
            'n_points': 1,
            'prompt': None,
            'answer': None,
            'answer_template': None,
        })
    
    print(f"Added {len(n1_stimuli)} position prompts for n=1 stimuli")
    
    # Step 2: Ensure each stimulus appears at least once
    # For stimuli not yet covered, add one random main query type
    covered_ids = set(prompt['id'] for prompt in selected_prompts)
    uncovered_stimuli = main_stimuli[~main_stimuli['id'].isin(covered_ids)]
    
    for _, stim in uncovered_stimuli.iterrows():
        # Choose appropriate query types based on n_points
        n_points = stim['n_points']
        if n_points == 1:
            valid_queries = ['count', 'position']
        elif n_points == 2:
            valid_queries = ['count', 'position', 'distance']
        else:
            valid_queries = MAIN_QUERY_TYPES
        
        # Pick random query type
        task_type = np.random.choice(valid_queries)
        
        selected_prompts.append({
            'id': stim['id'],
            'metadata_path': stim['metadata_path'],
            'task_type': task_type,
            'n_points': n_points,
            'prompt': None,
            'answer': None,
            'answer_template': None,
        })
    
    print(f"Added {len(uncovered_stimuli)} prompts to ensure each stimulus appears at least once")
    
    # Step 3: Fill remaining prompts with balanced sampling
    remaining_prompts = target_total - len(selected_prompts)
    
    if remaining_prompts > 0:
        # Get unique n_points values
        unique_n_points = sorted(main_stimuli['n_points'].unique())
        
        # Calculate how many prompts per n_points and query type combination
        valid_combinations = []
        for n_points in unique_n_points:
            n_point_stimuli = main_stimuli[main_stimuli['n_points'] == n_points]
            
            if n_points == 1:
                valid_queries = ['count']  # position already covered
            elif n_points == 2:
                valid_queries = ['count', 'distance']  # position covered in step 2
            else:
                valid_queries = MAIN_QUERY_TYPES
            
            for query_type in valid_queries:
                valid_combinations.append((n_points, query_type, len(n_point_stimuli)))
        
        # Balance across combinations
        prompts_per_combo = remaining_prompts // len(valid_combinations)
        extra_prompts = remaining_prompts % len(valid_combinations)
        
        for i, (n_points, query_type, available_stimuli) in enumerate(valid_combinations):
            combo_prompts = prompts_per_combo + (1 if i < extra_prompts else 0)
            combo_prompts = min(combo_prompts, available_stimuli)  # Don't exceed available stimuli
            
            # Sample stimuli for this combination
            n_point_stimuli = main_stimuli[main_stimuli['n_points'] == n_points]
            sampled_stimuli = n_point_stimuli.sample(n=combo_prompts, replace=True)
            
            for _, stim in sampled_stimuli.iterrows():
                selected_prompts.append({
                    'id': stim['id'],
                    'metadata_path': stim['metadata_path'],
                    'task_type': query_type,
                    'n_points': n_points,
                    'prompt': None,
                    'answer': None,
                    'answer_template': None,
                })
        
        print(f"Added {remaining_prompts} balanced prompts across query types and n_points")
    
    return pd.DataFrame(selected_prompts)


def generate_ensemble_prompts(ensemble_stimuli: pd.DataFrame) -> pd.DataFrame:
    """
    Generate prompts for ensemble stimuli
    Each stimulus gets exactly one prompt based on its stimulus_type
    """
    selected_prompts = []
    
    for _, stim in ensemble_stimuli.iterrows():
        # Determine task type from generator_metadata
        gen_metadata = stim['generator_metadata']
        stimulus_type = gen_metadata['stimulus_type']
        
        if stimulus_type in ENSEMBLE_QUERY_TYPES:
            selected_prompts.append({
                'id': stim['id'],
                'metadata_path': stim['metadata_path'],
                'task_type': stimulus_type,
                'n_points': stim.get('n_points', None),
                'prompt': None,
                'answer': None,
                'answer_template': None,
            })
    
    print(f"Added {len(selected_prompts)} ensemble prompts")
    return pd.DataFrame(selected_prompts)


def generate_prompt(row: pd.Series) -> Tuple[str, str, List]:
    """
    Generate prompt based on stimulus metadata and task type
    """
    metadata = np.load(row['metadata_path'], allow_pickle=True).item() if isinstance(row['metadata_path'], str) else row['metadata_path']
    task = row['task_type']

    if "line" in row['metadata_path']:
        queries = LINE_QUERIES
    elif "bar" in row['metadata_path']:
        queries = BAR_QUERIES
    else:
        queries = QUERIES

    # Handle ensemble queries
    if task in ENSEMBLE_QUERY_TYPES:
        return generate_ensemble_prompt(row)

    # Handle main queries
    if 'grid_points' in metadata:
        grid_points = metadata['grid_points']
        relative_points = metadata['relative_points']
        rounded_relative_points = metadata['rounded_relative_points']
    else:
        grid_points = metadata['points']
        relative_points = metadata['points']
        rounded_relative_points = metadata['points']
    if task == 'count':
        prompt = queries['count']['prompt']
        answer_template = queries['count']['answer_template']
        answer = [len(grid_points)]
    
    elif task == 'position':
        # Generate position task prompt
        obj_idx = np.random.randint(0, len(grid_points))
        target = describe_datapoint(obj_idx, metadata)
        prompt = queries['position']['prompt'].format(target=target)
        answer_template = queries['position']['answer_template'].format(target=target)
        answer = [rounded_relative_points[obj_idx]]

    elif task == 'distance':
        # Generate distance task prompt
        obj1_idx, obj2_idx = np.random.choice(len(grid_points), size=2, replace=False)
        target1 = describe_datapoint(obj1_idx, metadata)
        target2 = describe_datapoint(obj2_idx, metadata)

        points = np.array(relative_points)
        prompt = queries['distance']['prompt'].format(target1=target1, target2=target2)
        answer_template = queries['distance']['answer_template'].format(target1=target1, target2=target2)
        answer = np.sqrt(np.sum((points[obj1_idx] - points[obj2_idx])**2))
        answer = [math.floor(answer), math.ceil(answer)]
        answer = [rounded_relative(answer[0], metadata), rounded_relative(answer[1], metadata)]
    
    elif task in ['min_x', 'max_x', 'min_y', 'max_y']:
        axis = task.split('_')[1]
        task_base = task.split('_')[0]
        points = np.array(grid_points)

        # Generate min/max task prompt
        prompt = queries[task_base]['prompt'].format(axis=axis)
        answer_template = queries[task_base]['answer_template'].format(axis=axis)
        min_max_val = np.min(points[:, 0] if axis == 'x' else points[:, 1]) if task_base == 'min' else np.max(points[:, 0] if axis == 'x' else points[:, 1])
        answer = np.where(points[:, 0] if axis == 'x' else points[:, 1] == min_max_val)[0]
        answer = [describe_datapoint(a, metadata) for a in answer]

    elif task == 'mean':
        prompt = queries[task]['prompt']
        answer_template = queries[task]['answer_template']
        points = np.array(relative_points)
        answer = np.mean(points, axis=0)
        answer = [
            (math.floor(answer[0]), math.floor(answer[1])),
            (math.floor(answer[0]), math.ceil(answer[1])),
            (math.ceil(answer[0]), math.floor(answer[1])),
            (math.ceil(answer[0]), math.ceil(answer[1]))
        ]
        answer = [
            (rounded_relative(t[0], metadata), rounded_relative(t[1], metadata)) for t in answer
        ]
    else:
        raise ValueError(f"Task type {task} not supported")
    
    return prompt, answer_template, answer


def generate_ensemble_prompt(row: pd.Series) -> Tuple[str, str, List]:
    """
    Generate prompt based on ensemble stimulus metadata and task type
    """
    metadata = np.load(row['metadata_path'], allow_pickle=True).item() if isinstance(row['metadata_path'], str) else row['metadata_path']
    task = row['task_type']
    
    # Get generator metadata which contains stimulus-specific information
    generator_metadata = metadata['generator_metadata']
    
    if task == 'correlation':
        prompt = QUERIES['correlation']['prompt']
        answer_template = QUERIES['correlation']['answer_template']
        r = generator_metadata['r']
        option_1 = np.random.choice(['stronger', 'weaker'])
        option_2 = 'weaker' if option_1 == 'stronger' else 'stronger'
        threshold = np.random.choice([0.25, 0.5, 0.75])
        if r < 0:
            threshold = -threshold
            answer = ['weaker' if r > threshold else 'stronger']
        else:
            answer = ['stronger' if r > threshold else 'weaker']

        prompt = prompt.format(option_1=option_1, option_2=option_2, threshold=threshold.item())
    
    elif task == 'cluster':
        # Get visual identities
        visual_identity = metadata['visual_identity']
        cluster1_color = visual_identity['cluster_1']['color'][0]  # Take first color as representative
        cluster2_color = visual_identity['cluster_2']['color'][0]
        query_color = visual_identity['query_point']['color']
        query_shape = SYMBOL_TO_SHAPE_MAP[visual_identity['query_point']['shape']]
        
        # Format prompt
        prompt = QUERIES['cluster']['prompt'].format(
            query_color=query_color,
            query_shape=query_shape,
            color1=cluster1_color,
            color2=cluster2_color
        )
        answer_template = QUERIES['cluster']['answer_template'].format(
            query_color=query_color,
            query_shape=query_shape
        )
        
        # Get answer from metadata
        query_cluster = generator_metadata['query_cluster']
        answer = [f"{cluster1_color} cluster" if query_cluster == 0 else f"{cluster2_color} cluster"]
    
    elif task == 'function':
        prompt = QUERIES['function']['prompt']
        answer_template = QUERIES['function']['answer_template']
        func_type = generator_metadata['function_type']
        answer = [func_type]

        option_1, option_2 = np.random.choice(['linear', 'quadratic', 'exponential', 'logarithmic'], size=2, replace=False)
        prompt = prompt.format(option_1=option_1, option_2=option_2)
    
    elif task == 'outlier':
        prompt = QUERIES['outlier']['prompt']
        answer_template = QUERIES['outlier']['answer_template']
        outlier_point = np.array(generator_metadata['outlier_point'])
        rel_outlier_point = to_relative(outlier_point, metadata)
        # Round to nearest integer
        rounded_point = rounded_relative(rel_outlier_point, metadata)
        answer = [(rounded_point[0], rounded_point[1])]

        stimulus_type = generator_metadata['stimulus_type']
        prompt = prompt.format(stimulus_type=stimulus_type)
    
    else:
        raise ValueError(f"Task type {task} not supported for ensemble stimuli")
    
    return prompt, answer_template, answer


def generate_prompts_from_config(config_path: str) -> pd.DataFrame:
    """
    Main function to generate balanced prompts based on config file
    
    Args:
        config_path: Path to YAML config file
    
    Returns:
        DataFrame with all generated prompts
    """
    # Load config
    config = load_config(config_path)
    root_dir = config['global']['root_dir']
    prompt_config = config.get('prompt_generation', {})
    
    print(f"Loading stimuli from: {root_dir}")
    
    # Load all stimuli metadata
    stimuli_df = load_stimuli_metadata(root_dir)
    print(f"Loaded {len(stimuli_df)} stimuli")
    
    # Separate main and ensemble stimuli
    main_stimuli, ensemble_stimuli = separate_stimuli(stimuli_df)
    print(f"Main stimuli: {len(main_stimuli)}, Ensemble stimuli: {len(ensemble_stimuli)}")
    
    # Check if generate_all is enabled
    generate_all = prompt_config.get('generate_all', False)
    
    if generate_all:
        print("Generating ALL possible prompts for all stimuli")
        all_prompts = generate_all_prompts(main_stimuli, ensemble_stimuli)
    else:
        # Generate balanced prompts based on config
        main_prompt_config = prompt_config.get('main_prompts', {})
        ensemble_prompt_config = prompt_config.get('ensemble_prompts', {})
        
        # Validate config has required sections
        if not main_prompt_config and len(main_stimuli) > 0:
            print("Warning: No main_prompts config found, but main stimuli exist. Using default settings.")
            main_prompt_config = {
                'targets': {'count': 100, 'position': 100, 'distance': 100, 'min': 100, 'max': 100, 'mean': 100},
                'balance_across': ['n_points', 'task_type']
            }
        
        if not ensemble_prompt_config and len(ensemble_stimuli) > 0:
            print("Warning: No ensemble_prompts config found, but ensemble stimuli exist. Using default settings.")
            ensemble_prompt_config = {
                'targets': {'correlation': 100, 'function': 100, 'cluster': 100, 'outlier': 100},
                'balance_across': ['n_points', 'task_type'],
                'sampling_strategy': 'one_per_stimulus'
            }
        
        main_prompts = pd.DataFrame()
        ensemble_prompts = pd.DataFrame()
        
        if len(main_stimuli) > 0 and main_prompt_config:
            main_prompts = generate_config_based_main_prompts(main_stimuli, main_prompt_config)
        
        if len(ensemble_stimuli) > 0 and ensemble_prompt_config:
            ensemble_prompts = generate_config_based_ensemble_prompts(ensemble_stimuli, ensemble_prompt_config)
        
        # Combine all prompts
        prompts_to_combine = [df for df in [main_prompts, ensemble_prompts] if len(df) > 0]
        if prompts_to_combine:
            all_prompts = pd.concat(prompts_to_combine, ignore_index=True)
        else:
            all_prompts = pd.DataFrame()
    
    # Generate actual prompt text, answer templates, and answers
    if len(all_prompts) > 0:
        print("Generating prompt text...")
        all_prompts[['prompt', 'answer_template', 'answer']] = all_prompts.apply(
            lambda row: pd.Series(generate_prompt(row)), axis=1
        )
    else:
        print("No prompts to generate - empty result set")
    
    # Print summary
    print("\nPrompt generation summary:")
    
    if generate_all:
        print("Generated ALL possible prompts:")
        task_summary = all_prompts.groupby(['task_type']).size()
        print(task_summary)
        
        if 'n_points' in all_prompts.columns:
            n_points_summary = all_prompts.groupby(['task_type', 'n_points']).size().unstack(fill_value=0)
            print("\nPrompts by task type and n_points:")
            print(n_points_summary)
    else:
        main_prompts_only = all_prompts[all_prompts['task_type'].isin(['count', 'position', 'distance', 'min_x', 'min_y', 'max_x', 'max_y', 'mean'])]
        ensemble_prompts_only = all_prompts[all_prompts['task_type'].isin(ENSEMBLE_QUERY_TYPES)]
        
        if len(main_prompts_only) > 0:
            print("Main prompts by task type:")
            main_summary = main_prompts_only.groupby(['task_type', 'n_points']).size().unstack(fill_value=0)
            print(main_summary)
        
        if len(ensemble_prompts_only) > 0:
            print("\nEnsemble prompts by task type:")
            ensemble_summary = ensemble_prompts_only.groupby(['task_type']).size()
            print(ensemble_summary)
    
    print(f"\nTotal prompts: {len(all_prompts)}")
    
    # Save prompts
    if len(all_prompts) > 0:
        output_path = Path(root_dir) / "stimuli_and_prompts.csv"
        all_prompts.to_csv(output_path, index=False)
        print(f"Saved prompts to: {output_path}")
    else:
        print("No prompts to save")
    
    return all_prompts


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML file')
    args = parser.parse_args()

    # Generate prompts from config
    all_prompts = generate_prompts_from_config(config_path=args.config)
