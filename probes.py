import os
import random
import glob
import itertools
from itertools import chain
from typing import Dict, Tuple, Callable, List, Union
import tqdm
from PIL import Image

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import wandb

from models.models import get_model_handler
from utils.generate_prompts import generate_prompt, SYMBOL_TO_SHAPE_MAP
from utils.stimuli import generate_segmentation_for_metadata
from utils.probes import create_bar_probe_dataset, create_line_probe_dataset


PROMPT_PREFIX = """
The image shows a scatter plot displaying the relationship between two quantitative variables, labeled 'x' (horizontal axis) and 'y' (vertical axis).
Each observation in the dataset is represented by a single graphical element (circle, triangle, square, or star) positioned on the coordinate plane according to its exact x- and y-values.
The data points appear in four distinct colors (red, green, blue, or orange).
Please answer the following question based on the information conveyed by this scatter plot.
"""
PROMPT_PREFIX_CONCISE = """
The image shows a scatter plot displaying the relationship between two quantitative variables, labeled 'x' (horizontal axis) and 'y' (vertical axis).
Each observation in the dataset is represented by a single graphical element (circle, triangle, square, or star) positioned on the coordinate plane according to its exact x- and y-values.
The data points appear in four distinct colors (red, green, blue, or orange).
Please answer the following question based on the information conveyed by this scatter plot.
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

class ProbeDataset(Dataset):
    def __init__(
        self, 
        features: Dict[str, torch.Tensor],
        labels: Dict[str, torch.Tensor],
    ):
        self.features = features
        self.labels = labels
        self.ids = list(features.keys())

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        stimulus_id = self.ids[idx]
        features = self.features[stimulus_id]
        if isinstance(features, torch.Tensor):
            features = features.to("cuda")  # Move to GPU when needed
        else:
            features = torch.tensor(features).to("cuda")
        return features, self.labels[stimulus_id], stimulus_id
        #return self.features[stimulus_id], self.labels[stimulus_id], stimulus_id
    
class LinearProbe(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.W = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        x = x.to(dtype=torch.float32)
        return self.W(x)
    

def _patchify_segmentation(seg):
    """
    Patchify a segmentation mask into a list of ViT patches.
    """
    num_patches = 560 // 14
    patches = []
    for i in range(num_patches):
        for j in range(num_patches):
            patch = seg[i*14:(i+1)*14, j*14:(j+1)*14]
            patches.append(patch)

    idxs = []
    for idx, patch in enumerate(patches):
        # Check if any pixel is not white (255)
        if np.any(patch == 1):
            idxs.append(idx + 1) 

    return idxs

def _process_segmentation(seg):
    """
    Recursively process a nested dictionary until reaching lists of arrays.
    
    Args:
        seg: Nested dictionary containing arrays
        patchify_fn: Function to apply to each array
    
    Returns:
        Dictionary with same structure but processed arrays
    """
    if isinstance(seg, dict):
        return {k: _process_segmentation(v) for k, v in seg.items()}
    elif isinstance(seg, list):
        return [_patchify_segmentation(arr) for arr in seg]
    else:
        raise ValueError(f"Unexpected type: {type(seg)}")

    
def load_full_dataset(
    data_root: str = "../../../../data/alexart/intervention_data",
) -> pd.DataFrame:
    """Load full dataset.

    Args:
        data_root: Root directory containing stimuli and metadata
    ) -> pd.DataFrame:
    """

    csv_path = os.path.join(data_root, "stimuli_and_prompts.csv")
    metadata_dir = os.path.join(data_root, "metadata")

    if os.path.exists(csv_path):
        dataset = pd.read_csv(csv_path)
        metadata = [
            np.load(os.path.join(metadata_dir, f"{row['id']}.npy"), allow_pickle=True).item()
            for _, row in dataset.iterrows()
        ]
        return pd.DataFrame(metadata)

    if not os.path.isdir(metadata_dir):
        raise FileNotFoundError(f"No dataset found in {data_root}; expected {csv_path} or {metadata_dir}")

    metadata_files = glob.glob(os.path.join(metadata_dir, "*.npy"))
    metadata = [np.load(f, allow_pickle=True).item() for f in metadata_files]
    metadata = pd.DataFrame(metadata)
    return metadata

"""
def aggregate_features(
    f, 
    metadata, 
    agg_function,
    segmentation_units=None,
    aggregate=True,
):
    def agg(feature):
        feature = feature.squeeze()
        if aggregate:
            agg_dims = list(range(len(feature.shape) - 1))
            feature = agg_function(feature, agg_dims)
            feature = feature.squeeze()
        return feature

    if segmentation_units is not None:
        if metadata["segmentation_path"] is None:
            generate_segmentation_for_metadata(metadata['metadata_path'])
            metadata['segmentation_path'] = metadata['image_path'].replace('/stimuli/', '/segmentation/').replace('.png', '.npy')
        if "segmentation_patchified_560x14" not in metadata:
            seg = np.load(metadata['segmentation_path'], allow_pickle=True).item()
            metadata["segmentation_patchified_560x14"] = _process_segmentation(seg)
            np.save(metadata['metadata_path'], metadata, allow_pickle=True)

        seg = metadata["segmentation_patchified_560x14"]
        all_feats = {m: [] for m in f.keys()}
        for unit in segmentation_units:
            if unit == "dots":
                # Find the dot differences between the two images
                #unit_tokens = list(set(chain(*seg['dots'])))
                unit_tokens = [list(tokens) for tokens in seg['dots']]
            elif unit == "cls":
                #unit_tokens = [0]
                unit_tokens = [[0]]
            elif unit == "bg":
                # Get all non-background tokens for source image
                num_tokens = (560 // 14) ** 2 + 1
                non_bg = set()
                for seg_type in ['dots', 'axis_x', 'axis_y', 'cls', 'ticks_x', 'ticks_y', 'tick_labels_x', 'tick_labels_y']:
                    if seg_type in seg:
                        non_bg.update(chain(*seg[seg_type]))
                #unit_tokens = list(set(range(num_tokens)) - non_bg)
                unit_tokens = [list(set(range(num_tokens)) - non_bg)]
            else:
                unit_tokens = []
                for axis in ["x", "y"]:
                    #unit_tokens.extend(list(set(chain(*seg[f"{unit}_{axis}"]))))
                    unit_tokens.extend([list(tokens) for tokens in seg[f"{unit}_{axis}"]])
            
            for m in f.keys():
                #agg_feats = agg(f[m].squeeze()[0, unit_tokens, :])
                agg_feats = [agg(f[m].squeeze()[0, tokens, :]) for tokens in unit_tokens if len(agg(f[m].squeeze()[0, tokens, :])) > 0]
                #all_feats[m].append(agg_feats)
                all_feats[m].extend(agg_feats)
        
        return {m: torch.stack(all_feats[m], dim=0).flatten() for m in f.keys()}
    else:
        for m in f.keys():
            f[m] = agg(f[m])
        return f
"""

def load_model_features(  # NOT USED
    task_stimuli: pd.DataFrame,
    full_dataset: pd.DataFrame,
    model_id: str,
    target_colors: Union[str, List[str]],
    target_shapes: Union[str, List[str]],
    layer: int,
    component: str,
    segmentation_units: List[str] = None,
    max_dots: int = 16,
    task: str = "position",
) -> Dict[str, torch.Tensor]:
    """Load or extract model features for a given set of stimuli.
    
    This function handles loading existing features from disk or extracting new features
    from a model for a given set of stimuli. It supports multiple modalities including
    vision, connector, and language features.
    
    Args:
        model_handler: Model handler instance for feature extraction
        stimuli: DataFrame containing stimulus information including IDs and image paths
        data_root: Root directory containing model features and stimuli
        modality: Type of features to extract ('vision', 'connector', or 'language')
        task: Task type for language features (e.g., 'position')
        aggregate: Whether to aggregate features (currently unused)
        
    Returns:
        Dictionary mapping stimulus IDs to their corresponding feature tensors
    """
    if target_colors == "all":
        target_colors = list(set(task_stimuli['color'].explode()))
        target_shapes = [SYMBOL_TO_SHAPE_MAP[shape] for shape in list(set(task_stimuli['plot_dot_shape'].explode()))]
    else:
        if isinstance(target_colors, str):
            target_colors = [target_colors]
            target_shapes = [target_shapes]
    
    # First, load all features and map them to their stimulus IDs
    feature_map = {}  # Maps stimulus ID to dict of features for each color/shape
    # First get feature dimension by loading one example
    sample_color, sample_shape = target_colors[0], target_shapes[0]
    if segmentation_units:
        sample_path = f"/data/alexart/features/{model_id.split('/')[-1]}/{sample_color}_{sample_shape}_{component.replace('.', '-').replace('_', '-')}_{layer}_{segmentation_units[0]}.npy"
    else:
        sample_path = f"/data/alexart/features/{model_id.split('/')[-1]}/{sample_color}_{sample_shape}_{component.replace('.', '-').replace('_', '-')}_{layer}_{task}.npy"
    sample_feat = np.load(sample_path, allow_pickle=True)[0]
    feat_dim = sample_feat.shape[-1]

    for target_color, target_shape in zip(target_colors, target_shapes):
        # Get the subset of full_dataset that contains this color/shape
        relevant_stimuli = []
        for _, row in full_dataset[full_dataset["point_generator"] == "grid"].iterrows():
            if pd.isna(row['segmentation_patchified_560x14']):
                continue
            shapes = [SYMBOL_TO_SHAPE_MAP[shape] for shape in row['plot_dot_shape']]
            colors = row['plot_color']
            objects = [f"{color} {shape}" for color, shape in zip(colors, shapes)]

            if f"{target_color} {target_shape}" in objects:
                relevant_stimuli.append(row)
        relevant_stimuli = pd.DataFrame(relevant_stimuli)

        # Load features for this color/shape combination
        if segmentation_units is not None:
            for unit in segmentation_units:
                try:
                    feat_path = f"/data/alexart/features/{model_id.split('/')[-1]}/{target_color}_{target_shape}_{component.replace('.', '-').replace('_', '-')}_{layer}_{unit}.npy"
                    feats = np.load(feat_path, allow_pickle=True)

                    # Map features to stimulus IDs
                    for i, stim_id in enumerate(relevant_stimuli['id']):
                        if stim_id not in feature_map:
                            feature_map[stim_id] = {}
                        if f"{target_color}_{target_shape}" not in feature_map[stim_id]:
                            feature_map[stim_id][f"{target_color}_{target_shape}"] = {}
                        feature_map[stim_id][f"{target_color}_{target_shape}"][unit] = feats[i]
                        
                except FileNotFoundError:
                    print(f"No features found for {model_id.split('/')[-1]} {target_color} {target_shape} {component} {layer} {unit}")
                    return None
        else:
            try:
                feat_path = f"/data/alexart/features/{model_id.split('/')[-1]}/{target_color}_{target_shape}_{component.replace('.', '-').replace('_', '-')}_{layer}_{task}.npy"
                feats = np.load(feat_path, allow_pickle=True)

                print(feats.shape)
                print(len(relevant_stimuli['id']))
                
                # Map features to stimulus IDs
                for i, stim_id in enumerate(relevant_stimuli['id']):
                    feature_map[stim_id] = feats[i].flatten()
            except FileNotFoundError:
                print(f"No features found for {model_id.split('/')[-1]} {target_color} {target_shape} {component} {layer}")
                return None

    # Now construct the final features for each task stimulus
    final_features = {}
    for idx, row in task_stimuli.iterrows():
        stim_id = row['id']
        if stim_id not in feature_map:
            continue
            
        # Get the colors and shapes for this stimulus
        colors = row['color']
        shapes = [SYMBOL_TO_SHAPE_MAP[shape] for shape in row['plot_dot_shape']]

        #"""
        if segmentation_units is not None:
            # Initialize arrays for dot features and other segmentation units
            dot_features = np.zeros((max_dots, feat_dim))  # Slots for individual dots
            other_unit_features = []  # For bg, axes, etc.
            
            # Fill in actual dot features
            dot_idx = 0
            for color, shape in zip(colors, shapes):
                key = f"{color}_{shape}"
                if key in feature_map[stim_id] and dot_idx < max_dots:
                    dot_features[dot_idx] = feature_map[stim_id][key]['dots']
                    dot_idx += 1
            
            # Add other segmentation unit features
            for unit in segmentation_units:
                if unit != 'dots':
                    unit_features = []
                    for color, shape in zip(colors, shapes):
                        key = f"{color}_{shape}"
                        if key in feature_map[stim_id]:
                            unit_features.append(feature_map[stim_id][key][unit])
                    if unit_features:
                        other_unit_features.append(np.mean(np.stack(unit_features, axis=0), axis=0))
            
            # Concatenate dot slots with averaged other unit features
            if other_unit_features:
                other_features = np.concatenate(other_unit_features, axis=-1)
                final_features[stim_id] = np.concatenate([dot_features.flatten(), other_features])
            else:
                final_features[stim_id] = dot_features.flatten()
        else:
            # Without segmentation units, just create slots for dots
            final_features[stim_id] = feature_map[stim_id]
        
        """
        # Collect all relevant features for this stimulus
        stim_features = []
        for color, shape in zip(colors, shapes):
            key = f"{color}_{shape}"
            if key in feature_map[stim_id]:
                if segmentation_units is not None:
                    unit_features = {unit: [] for unit in segmentation_units}

                    # Collect features for each color/shape combination
                    for color, shape in zip(colors, shapes):
                        key = f"{color}_{shape}"
                        if key in feature_map[stim_id]:
                            # Add features to appropriate unit lists
                            for unit in segmentation_units:
                                unit_features[unit].append(feature_map[stim_id][key][unit])

                    # Average features for each unit and concatenate
                    stim_features = []
                    for unit in segmentation_units:
                        if unit_features[unit]:  # Check if we have any features for this unit
                            # Stack along first dimension and take mean
                            unit_avg = np.mean(np.stack(unit_features[unit], axis=0), axis=0)
                            stim_features.append(unit_avg)
                    
                    if stim_features:  # If we have any features
                        final_features[stim_id] = np.concatenate(stim_features, axis=-1)
                else:
                    # Without segmentation units, just concatenate all features
                    stim_features = []
                    for color, shape in zip(colors, shapes):
                        key = f"{color}_{shape}"
                        if key in feature_map[stim_id]:
                            stim_features.append(feature_map[stim_id][key])
                    
                    if stim_features:
                        final_features[stim_id] = np.mean(np.stack(stim_features, axis=0), axis=0)
        
        if stim_features:
            # Concatenate all features for this stimulus
            final_features[stim_id] = np.concatenate(stim_features, axis=-1)
        """

    return final_features

    """
    all_feats = []

    # Handle single target case
    if isinstance(target_colors, str):
        if target_colors == "all":
            target_colors = list(set(stimuli['color'].explode()))
            target_shapes = [SYMBOL_TO_SHAPE_MAP[shape] for shape in list(set(stimuli['plot_dot_shape'].explode()))]
        else:
            target_colors = [target_colors]
            target_shapes = [target_shapes]
    
    for target_color, target_shape in zip(target_colors, target_shapes):
        if segmentation_units is not None:
            for unit in segmentation_units:
                try:
                    feat_path = f"../../../../data/alexart/features/{model_id.split('/')[-1]}/{target_color}_{target_shape}_{component.replace('.', '-').replace('_', '-')}_{layer}_{unit}.npy"
                    feats = np.load(feat_path, allow_pickle=True)
                    print("unit", unit, feats.shape)
                    all_feats.append(feats)
                except FileNotFoundError:
                    print(f"No features found for {model_id.split('/')[-1]} {target_color} {target_shape} {component} {layer} {unit}")
                    return None
        else:
            try:
                feat_path = f"../../../../data/alexart/features/{model_id.split('/')[-1]}/{target_color}_{target_shape}_{component.replace('.', '-').replace('_', '-')}_{layer}.npy"
                feats = np.load(feat_path, allow_pickle=True)
                all_feats.append(feats)
            except FileNotFoundError:
                print(f"No features found for {model_id.split('/')[-1]} {target_color} {target_shape} {component} {layer}")
                return None

    print("all_feats", np.concatenate(all_feats, axis=1).shape)
    return np.concatenate(all_feats, axis=1)
    """

def new_load_model_features(stimuli, model_id, component, layer, task, target_color=None, target_shape=None):
    model_features = {}

    component_name = component.replace('.', '-').replace('_', '-')

    get_features = []

    for _, row in stimuli.iterrows():
        sid = row['id']

        if "language" in component:
            feat_path = f"/data/alexart/features/{model_id.split('/')[-1]}/{component_name}/{layer}/{task}/{sid}.npy"
        else:
            feat_path = f"/data/alexart/features/{model_id.split('/')[-1]}/{component_name}/{layer}/{sid}.npy"

        try:
            feat = np.load(feat_path, allow_pickle=True)
            model_features[sid] = feat
        except FileNotFoundError:
            get_features.append((sid, row['image_path']))

    if len(get_features) > 0:
        model = get_model_handler(model_id, device="auto")
        if "fugu" in model_id.lower():
            model.model.load_adapter("/data/alexart/fugu/llama/checkpoint-660", adapter_name="custom_adapter")
            model.model.set_adapter("custom_adapter")
            print(f"Loaded adapter from /data/alexart/fugu/llama/checkpoint-660")

        """
        if "language" in component:
            if "llama" in model_id.lower():
                layers = list(range(40))
            elif "internvl" in model_id.lower():
                layers = list(range(47))
            elif "llava" in model_id.lower():
                layers = list(range(28))
        else:
            if "llama" in model_id.lower():
                layers = [3, 7, 15, 23, 30, 31]
                feature_dim = 1280
            elif "internvl" in model_id.lower():
                layers = [23]
                feature_dim = 1024
            elif "llava" in model_id.lower():
                layers = [25]
                feature_dim = 1152
        """
        layers = [layer]

        with torch.no_grad():
            for sid, image_path in tqdm.tqdm(get_features, desc="Extracting features"):
                image = Image.open(image_path)

                get_layers = []
                for l in layers:
                    if "language" in component:
                        feat_path = f"/data/alexart/features/{model_id.split('/')[-1]}/{component_name}/{l}/{task}/{sid}.npy"
                        os.makedirs(f"/data/alexart/features/{model_id.split('/')[-1]}/{component_name}/{l}/{task}", exist_ok=True)
                    else:
                        feat_path = f"/data/alexart/features/{model_id.split('/')[-1]}/{component_name}/{l}/{sid}.npy"
                        os.makedirs(f"/data/alexart/features/{model_id.split('/')[-1]}/{component_name}/{l}", exist_ok=True)
                    try:
                        _ = np.load(feat_path, allow_pickle=True)
                    except FileNotFoundError:
                        get_layers.append(l)

                if "language" in component:
                    if task == "position":
                        prompt = f"What is the (x, y) coordinate of the {target_color[0]} {target_shape[0]}?"
                    
                    activations = model.extract_language_features(prompt, image, layers=get_layers).squeeze(1)
                    for l in get_layers:
                        feat_path = f"/data/alexart/features/{model_id.split('/')[-1]}/{component_name}/{l}/{task}/{sid}.npy"

                        #features = activations[l].to(dtype=torch.float32).squeeze().unsqueeze(0).detach().cpu().numpy()
                        features = activations.to(dtype=torch.float32).squeeze().unsqueeze(0).detach().cpu().numpy()

                        model_features[sid] = features
                        np.save(feat_path, features, allow_pickle=True)

                else:
                    activations = model.extract_visual_features(image).squeeze()

                    if "llama" in model_id.lower():
                        activations = activations[0]
                    for l in get_layers:
                        layer_idx = layers.index(l)
                        if "llama" in model_id.lower():
                            features = activations[:, layer_idx*feature_dim:(layer_idx+1)*feature_dim].to(dtype=torch.float32).detach().cpu().numpy()
                        else:
                            features = activations.to(dtype=torch.float32).detach().cpu().numpy()
                        feat_path = f"/data/alexart/features/{model_id.split('/')[-1]}/{component_name}/{l}/{sid}.npy"
                        model_features[sid] = features
                        np.save(feat_path, features, allow_pickle=True)

        del model
        torch.cuda.empty_cache()
        import gc; gc.collect()

    for sid in model_features.keys():
        model_features[sid] = model_features[sid].flatten()
    
    return model_features
    
def load_and_process_data(
    model_id: str,
    model_handler_device: str,
    task: str,
    target_color: Union[str, List[str]] = "green",  # Add parameters for target object
    target_shape: Union[str, List[str]] = "triangle",
    *,
    chart_type: str = "scatter",
    target_x_value: int | None = None,
    layer: int = 0,
    component: str = "vision.block_output",
    generalization: str = "cs",  # "cs" or "pos"
    modality: str = "vision",
    data_root: str = "/data/alexart/intervention_data",
    train_size: float = 0.8,
    segmentation_units: List[str] = None,
) -> Tuple[ProbeDataset, ProbeDataset]:
    """Load and process data for probe training.
    
    Args:
        n_points: Number of points in the stimulus
        task: Task to use for probe training
    ): -> Tuple[ProbeDataset, ProbeDataset]:
    """
    # For other tasks, convert single target to list format
    if target_color == "all" or target_shape == "all":
        target_color = "all"
        target_shape = "all"
        target_objects = "all"
    elif isinstance(target_color, str):
        target_color = [target_color]
        target_shape = [target_shape]
        target_objects = [f"{c} {s}" for c, s in zip(target_color, target_shape)]
    else:
        raise ValueError("Invalid target color or shape")

    full_dataset = load_full_dataset(data_root)
    #stimuli = full_dataset[full_dataset["point_generator"] == "grid"]
    stimuli = []
    for _, row in full_dataset[full_dataset["point_generator"] == "grid"].iterrows():
        #if pd.isna(row['segmentation_patchified_560x14']):
        #    continue
        shapes = [SYMBOL_TO_SHAPE_MAP[shape] for shape in row['plot_dot_shape']]
        colors = row['plot_color']
        objects = [f"{color} {shape}" for color, shape in zip(colors, shapes)]

        if task == "position":
            if chart_type == "bar":
                if target_color[0] in colors:
                    stimuli.append(row)
            elif chart_type == "line":
                point_x = [int(pt[0]) for pt in row['points']]
                if target_x_value in point_x:
                    stimuli.append(row)
            else:
                if target_objects[0] in objects:
                    stimuli.append(row)

        else:
            # For other tasks (count, min/max, mean), include all images
            # We'll get features for all dots
            if task == "count":
                stimuli.append(row)
            elif len(objects) >= 4:
                stimuli.append(row)
    
    stimuli = pd.DataFrame(stimuli)

    model_feats = new_load_model_features(
        stimuli,
        model_id,
        component,
        layer,
        task,
        target_color,
        target_shape
    )

    """
    model_feats = load_model_features(
        stimuli,
        full_dataset,
        model_id,
        target_color,
        target_shape,
        layer,
        component,
        segmentation_units=segmentation_units
    )
    if model_feats is None:
        return None, None
    model_feats = {id: torch.tensor(model_feats[id]).to("cuda") for id in model_feats.keys()}
    #model_feats = {id: torch.tensor(model_feats[i]).to("cuda") for i, id in enumerate(stimuli['id'])}
    """
    for k, v in model_feats.items():
        model_feats[k] = torch.tensor(v)

    labels = {}

    for i, row in stimuli.iterrows():
        labels[row['id']] = {}

        if task == "position":
            # Sort points left to right, top down
            points = row['points']
            colors = row['plot_color']
            shapes = [SYMBOL_TO_SHAPE_MAP[shape] for shape in row['plot_dot_shape']]
            objects = [f"{color} {shape}" for color, shape in zip(colors, shapes)]
            if chart_type == "bar":
                target_object_idx = colors.index(target_color[0])
                point = points[target_object_idx]
            elif chart_type == "line":
                target_object_idx = [int(pt[0]) for pt in points].index(target_x_value)
                point = points[target_object_idx]
            else:
                target_object = f"{target_color[0]} {target_shape[0]}"
                target_object_idx = objects.index(target_object)
                point = points[target_object_idx]

            x_pos = int(point[0]) - 1  # Convert to 0-based index
            y_pos = int(point[1]) - 1  # Convert to 0-based index
            
            # Create one-hot vectors for this point
            x_one_hot = np.zeros(8)
            y_one_hot = np.zeros(8)
            
            x_one_hot[x_pos] = 1
            y_one_hot[y_pos] = 1
            
            # Store in dictionary
            labels[row['id']]['x'] = x_one_hot  #np.stack(x_positions)
            labels[row['id']]['y'] = y_one_hot  #np.stack(y_positions)
            labels[row['id']]['task_x'] = x_one_hot
            labels[row['id']]['task_y'] = y_one_hot

        elif task == "count":
            one_hot = np.zeros(6)
            unique_n = list(stimuli['n_points'].unique())
            one_hot[unique_n.index(row['n_points'])] = 1
            labels[row['id']]['task'] = one_hot
    

        elif task == "min_x":
            one_hot = np.zeros(8)
            idx = np.array(row['points'])[:, 0].min()
            one_hot[idx - 1] = 1
            labels[row['id']]['task'] = one_hot
        
        elif task == "min_y":
            one_hot = np.zeros(8)
            idx = np.array(row['points'])[:, 1].min()
            one_hot[idx - 1] = 1
            labels[row['id']]['task'] = one_hot
        
        elif task == "max_x":
            one_hot = np.zeros(8)
            idx = np.array(row['points'])[:, 0].max()
            one_hot[idx - 1] = 1
            labels[row['id']]['task'] = one_hot
        
        elif task == "max_y":
            one_hot = np.zeros(8)
            idx = np.array(row['points'])[:, 1].max()
            one_hot[idx - 1] = 1
            labels[row['id']]['task'] = one_hot

        elif task == "mean":
            # Create one-hot vectors
            x_one_hot = np.zeros(8)
            y_one_hot = np.zeros(8)

            x_mean = np.array(row['points'])[:, 0].mean()
            y_mean = np.array(row['points'])[:, 1].mean()

            x_mean = round(x_mean)
            y_mean = round(y_mean)

            x_one_hot[x_mean - 1] = 1
            y_one_hot[y_mean - 1] = 1

            labels[row['id']]['task_x'] = x_one_hot
            labels[row['id']]['task_y'] = y_one_hot

    if generalization == "pos":
        if task == "position":
            ## Position generalization 
            all_positions = list(set(tuple(point) for point in stimuli['points'].explode()))
            unique_x = sorted(set(x for x, _ in all_positions))
            unique_y = sorted(set(y for _, y in all_positions))

            # Create a dictionary of positions grouped by x coordinate
            positions_by_x = {x: set(pos[1] for pos in all_positions if pos[0] == x) for x in unique_x}

            # For each x, we'll select one y to be in the test set
            # We'll rotate which y is selected to ensure each y appears in test set exactly once
            test_positions = set()
            train_positions = set()

            # Create a list of available y values to ensure each is used exactly once
            available_y_values_1 = unique_y.copy()
            available_y_values_2 = unique_y.copy()
            random.shuffle(available_y_values_1)
            random.shuffle(available_y_values_2)

            # For each x coordinate, select a random y from remaining available y values
            for x in unique_x:
                # Get first y value
                test_y1 = available_y_values_1.pop()
                test_positions.add((x, test_y1))
                
                # Try to get a second y value that's different
                max_attempts = 5  # Limit the number of attempts to avoid infinite loops
                attempts = 0
                found_second_y = False

                while available_y_values_2 and attempts < max_attempts:
                    test_y2 = available_y_values_2.pop()
                    if test_y2 != test_y1:
                        test_positions.add((x, test_y2))
                        found_second_y = True
                        break
                    else:
                        # Put this y back in the pool
                        available_y_values_2.append(test_y2)
                        random.shuffle(available_y_values_2)
                        attempts += 1
                
                # Add all other positions with this x to train set
                for y in positions_by_x[x]:
                    if y != test_y1 and (not found_second_y or y != test_y2):
                        train_positions.add((x, y))

            # Convert positions to lists for consistency with rest of code
            train_positions = list(train_positions)
            test_positions = list(test_positions)

            # Split the data into train and test sets
            train_ids = []
            test_ids = []

            for idx, row in stimuli.iterrows():
                pos = (np.argmax(labels[row['id']]['x']) + 1, np.argmax(labels[row['id']]['y']) + 1)
                if pos in train_positions:
                    train_ids.append(row['id'])
                else:
                    test_ids.append(row['id'])
        else:
            """
            # For non-position tasks, just do randogit m split
            valid_ids = list(model_feats.keys())
            random.shuffle(valid_ids)
            split_idx = int(len(valid_ids) * 0.8)  # 80-20 split
            train_ids = valid_ids[:split_idx]
            test_ids = valid_ids[split_idx:]
            """
            # Get valid IDs that have features
            valid_ids = set(model_feats.keys())

            # Get all possible positions from valid stimuli only
            all_positions = list(set(
                tuple(point) 
                for idx, row in stimuli.iterrows() 
                if row['id'] in valid_ids
                for point in row['points']
            ))
            unique_x = sorted(set(x for x, _ in all_positions))
            unique_y = sorted(set(y for _, y in all_positions))

            # Create train/test position splits as before
            positions_by_x = {x: set(pos[1] for pos in all_positions if pos[0] == x) for x in unique_x}
            test_positions = set()
            train_positions = set()

            # Select test positions (same logic as before)
            available_y_values_1 = unique_y.copy()
            available_y_values_2 = unique_y.copy()
            random.shuffle(available_y_values_1)
            random.shuffle(available_y_values_2)

            for x in unique_x:
                test_y1 = available_y_values_1.pop()
                test_positions.add((x, test_y1))
                
                max_attempts = 5
                attempts = 0
                found_second_y = False

                while available_y_values_2 and attempts < max_attempts:
                    test_y2 = available_y_values_2.pop()
                    if test_y2 != test_y1:
                        test_positions.add((x, test_y2))
                        found_second_y = True
                        break
                    else:
                        available_y_values_2.append(test_y2)
                        random.shuffle(available_y_values_2)
                        attempts += 1
                
                # Add remaining positions to train set
                for y in positions_by_x[x]:
                    if y != test_y1 and (not found_second_y or y != test_y2):
                        train_positions.add((x, y))

            # Split stimuli based on object positions, but only from valid IDs
            train_ids = []
            test_ids = []

            for idx, row in stimuli.iterrows():
                if row['id'] not in valid_ids:
                    continue
                    
                points = row['points']
                # Check if ANY point in the stimulus is in a test position
                has_test_position = any(tuple(point) in test_positions for point in points)
                # Check if ALL points are in valid positions (either train or test)
                all_points_valid = all(tuple(point) in (train_positions | test_positions) for point in points)
                
                if has_test_position and all_points_valid:
                    # If at least one point is in test position and all points are valid,
                    # put in test set
                    test_ids.append(row['id'])
                elif not has_test_position and all_points_valid:
                    # If no test positions and all points are valid,
                    # put in train set
                    train_ids.append(row['id'])

    elif generalization == "task":
        if task == "position":
            raise ValueError("Task generalization not applicable for position task")
        
        # Split based on task values
        if task == "count":
            unique_values = sorted(stimuli['n_points'].unique())
        elif task in ["min_x", "max_x", "min_y", "max_y"]:
            unique_values = sorted(stimuli[task].unique())
        elif task == "mean":
            # Bin mean x and y values separately
            x_means = [label['x'] for label in labels.values()]
            y_means = [label['y'] for label in labels.values()]
            unique_values = list(zip(
                np.percentile(x_means, np.linspace(0, 100, 4)),
                np.percentile(y_means, np.linspace(0, 100, 4))
            ))
        else:
            raise ValueError(f"Invalid task: {task}")
        
        # Select train values
        n_train = int(len(unique_values) * train_size)
        train_values = np.random.choice(unique_values, size=n_train, replace=False)
        
        # Create train/test split
        if task == "mean":
            train_mask = stimuli.apply(lambda row: 
                (np.mean(row['points'][:, 0]), np.mean(row['points'][:, 1])) in train_values, axis=1)
        else:
            train_mask = stimuli[task].isin(train_values)
        
        train_ids = list(stimuli[train_mask]['id'])
        test_ids = list(stimuli[~train_mask]['id'])

    else:
        raise ValueError(f"Invalid generalization: {generalization}")

    train_feats = {id: model_feats[id] for id in train_ids}
    test_feats = {id: model_feats[id] for id in test_ids}

    train_dataset = ProbeDataset(train_feats, labels)
    test_dataset = ProbeDataset(test_feats, labels)

    return train_dataset, test_dataset

def train_probe(
    model_id: str,
    modality: str,  # "vision", "connector", "language"
    task: str,  # "position", "count", "min_x", "min_y", "max_x", "max_y", "mean"
    generalization: str,  # "cs", "pos"
    target_color: str,
    target_shape: str,
    *,
    chart_type: str = "scatter",
    target_x_value: int | None = None,
    layer: int,
    component: str,
    data_root: str = "../../../../data/alexart/intervention_data",
    batch_size: int = 4,
    learning_rate: float = 1e-4,
    n_epochs: int = 2,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    model_handler_device: str = "cpu",
    segmentation_units: List[str] = None,
    scheduler_type: str = "reduce_on_plateau",
):
    """Train a probe on the specified task."""
    torch.cuda.empty_cache()

    # Load and process data
    train_dataset, test_dataset = load_and_process_data(
        model_id,
        model_handler_device,
        task,
        target_color,
        target_shape,
        chart_type=chart_type,
        target_x_value=target_x_value,
        layer=layer,
        component=component,
        generalization=generalization,
        modality=modality,
        data_root=data_root,
        segmentation_units=segmentation_units,
    )
    if train_dataset is None or test_dataset is None:
        return None

    feature_dim = next(iter(train_dataset.features.values())).shape[-1]
    k = list(train_dataset.labels.keys())[0]

    if task == "position" or task == "mean":
        label_dim = train_dataset.labels[k]["task_x"].shape[-1]*2
        index_dim = label_dim // 2
    else:
        label_dim = train_dataset.labels[k]["task"].shape[-1]

    # Initialize wandb
    config = {
        "model": model_id,
        "component": component,
        "segmentation_units": segmentation_units,
        "task": task,
        "chart_type": chart_type,
        "generalization": generalization,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "n_epochs": n_epochs,
        "label_dim": label_dim,
        "feature_dim": feature_dim,
    }
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    probe = LinearProbe(feature_dim, label_dim).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=learning_rate)

    # Choose scheduler based on type
    if scheduler_type == "reduce_on_plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=50,
            verbose=False,
        )
    elif scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=n_epochs,
            eta_min=1e-6
        )
    elif scheduler_type == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=30,
            gamma=0.1
        )
    else:
        scheduler = None

    criterion = nn.CrossEntropyLoss()

    if segmentation_units is not None:
        seg_str = "_".join(segmentation_units)
        seg_str = f"_{seg_str}"
    else:
        seg_str = ""

    target_descriptor = target_x_value if chart_type == "line" else target_color
    run_name = f"component-{component.replace('.', '-').replace('_', '-')}_task-{task}_target-{target_descriptor}_chart-{chart_type}_layer-{layer}{seg_str}"
    wandb.init(project=f"vlm-isomorphic-probing-{model_id.split('/')[-1]}", config=config, name=run_name)

    for epoch in tqdm.tqdm(range(n_epochs), desc=f"Training ({target_color} {target_shape}, layer {layer})..."):
        probe.train()
        train_loss = 0
        train_acc = 0

        for batch_idx, (features, labels, ids) in enumerate(train_loader):
            features = features.to(device)
            #if task == "position" or task == "mean":
            batch_size = features.shape[0]

            optimizer.zero_grad()

            outputs = probe(features)
            if task == "position" or task == "mean":
                outputs_x = outputs[:, :index_dim]
                outputs_y = outputs[:, index_dim:]

                # Get labels
                labels_x = labels['task_x'].to(device)  # Shape: (batch_size, n_points, 8)
                labels_y = labels['task_y'].to(device)  # Shape: (batch_size, n_points, 8)

                # Compute loss for each position prediction
                loss = 0
                loss += criterion(outputs_x, torch.argmax(labels_x, dim=1))
                loss += criterion(outputs_y, torch.argmax(labels_y, dim=1))

                # Compute accuracy
                preds_x = torch.argmax(outputs_x, dim=1)  # Shape: (batch_size, n_points)
                preds_y = torch.argmax(outputs_y, dim=1)
                labels_x_idx = torch.argmax(labels_x, dim=1)
                labels_y_idx = torch.argmax(labels_y, dim=1)

                # Point is correct if both x and y are correct
                correct = (preds_x == labels_x_idx) & (preds_y == labels_y_idx)
                train_acc += correct.float().mean().item()
            else:
                # Get labels
                labels = labels['task'].to(device) 

                # Compute loss for each position prediction
                loss = criterion(outputs, torch.argmax(labels, dim=1))

                # Compute accuracy
                preds = torch.argmax(outputs, dim=1)
                labels_idx = torch.argmax(labels, dim=1)

                # Point is correct if both x and y are correct
                correct = (preds == labels_idx)
                train_acc += correct.float().mean().item()

        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        # Evaluation
        probe.eval()
        test_loss = 0
        test_acc = 0
        with torch.no_grad():
            for batch_idx, (features, labels, ids) in enumerate(test_loader):
                features = features.to(device)
                #if task == "position" or task == "mean":
                batch_size = features.shape[0]

                outputs = probe(features)
                if task == "position" or task == "mean":
                    outputs_x = outputs[:, :index_dim]
                    outputs_y = outputs[:, index_dim:]

                    # Get labels
                    labels_x = labels['task_x'].to(device)  # Shape: (batch_size, n_points, 8)
                    labels_y = labels['task_y'].to(device)  # Shape: (batch_size, n_points, 8)

                    # Compute loss for each position prediction
                    loss = 0
                    loss += criterion(outputs_x, torch.argmax(labels_x, dim=1))
                    loss += criterion(outputs_y, torch.argmax(labels_y, dim=1))
                    
                    test_loss += loss.item()

                    # Compute accuracy
                    preds_x = torch.argmax(outputs_x, dim=1)
                    preds_y = torch.argmax(outputs_y, dim=1)
                    labels_x_idx = torch.argmax(labels_x, dim=1)
                    labels_y_idx = torch.argmax(labels_y, dim=1)

                    # Point is correct if both x and y are correct
                    correct = (preds_x == labels_x_idx) & (preds_y == labels_y_idx)
                    test_acc += correct.float().mean().item()
                else:
                    labels = labels['task'].to(device)

                    loss = criterion(outputs, torch.argmax(labels, dim=1))
                    test_loss += loss.item()

                    preds = torch.argmax(outputs, dim=1)
                    labels_idx = torch.argmax(labels, dim=1)

                    # Point is correct if both x and y are correct
                    correct = (preds == labels_idx)
                    test_acc += correct.float().mean().item()

        avg_train_loss = train_loss / len(train_loader)
        avg_train_acc = train_acc / len(train_loader)
        avg_test_loss = test_loss / len(test_loader)
        avg_test_acc = test_acc / len(test_loader)

        # Step the scheduler
        if scheduler is not None:
            if scheduler_type == "reduce_on_plateau":
                scheduler.step(avg_test_acc)
            else:
                scheduler.step()

        # Log metrics
        wandb.log({
            "train_loss": avg_train_loss,
            "train_acc": avg_train_acc,
            "test_loss": avg_test_loss,
            "test_acc": avg_test_acc,
            "epoch": epoch,
            "lr": optimizer.param_groups[0]['lr']
        })
    
    wandb.finish()

    probe.eval()
    if task == "position" or task == "mean":
        test_results = {
            'stimulus_ids': [],
            'true_x': [],
            'true_y': [],
            'pred_x': [],
            'pred_y': [],
        }
    else:
        test_results = {
            'stimulus_ids': [],
            'true': [],
            'pred': [],
        }
    
    with torch.no_grad():
        for batch_idx, (features, labels, ids) in enumerate(test_loader):
            features = features.to(device)
            outputs = probe(features)

            if task == "position" or task == "mean":
                outputs_x = outputs[:, :index_dim]
                outputs_y = outputs[:, index_dim:]
                
                labels_x = labels['task_x'].to(device)
                labels_y = labels['task_y'].to(device)
                
                preds_x = torch.argmax(outputs_x, dim=1).cpu().numpy()
                preds_y = torch.argmax(outputs_y, dim=1).cpu().numpy()
                true_x = torch.argmax(labels_x, dim=1).cpu().numpy()
                true_y = torch.argmax(labels_y, dim=1).cpu().numpy()
                
                test_results['stimulus_ids'].extend(ids)
                test_results['true_x'].extend(true_x)
                test_results['true_y'].extend(true_y)
                test_results['pred_x'].extend(preds_x)
                test_results['pred_y'].extend(preds_y)
            else:
                # Get labels
                labels = labels['task'].to(device) 

                # Compute accuracy
                pred = torch.argmax(outputs, dim=1)
                true = torch.argmax(labels, dim=1)

                test_results['stimulus_ids'].extend(ids)
                test_results['true'].extend(true)
                test_results['pred'].extend(pred)

    # Save test results
    os.makedirs("/data/alexart/probe_results", exist_ok=True)
    result_target = target_x_value if chart_type == "line" else target_color
    run_name = f"probing_{model_id.split('/')[-1]}_{component.replace('.', '-').replace('_', '-')}_{task}_{result_target}_{chart_type}_layer{layer}"
    results_path = f"/data/alexart/probe_results/{run_name}.npy"
    np.save(results_path, test_results)

    return probe

def _dataset_exists(dataset_root: str) -> bool:
    metadata_dir = os.path.join(dataset_root, "metadata")
    return os.path.isdir(metadata_dir) and len(glob.glob(os.path.join(metadata_dir, "*.npy"))) > 0


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id",
                        type=str,
                        required=True,
                        choices=["meta-llama/Llama-3.2-11B-Vision-Instruct",
                                "Salesforce/blip-vqa-base",
                                "allenai/Molmo-7B-D-0924",
                                "allenai/Molmo-7B-O-0924",
                                "facebook/chameleon-7b",
                                "OpenGVLab/InternVL3-14B-hf",
                                "llava-hf/llava-onevision-qwen2-7b-ov-hf",
                                "fugu/Llama-3.2-11B-Vision-Instruct"])
    parser.add_argument("--n_points", type=int, default=4)
    parser.add_argument("--layer", type=int, default=[-1], nargs="+")
    parser.add_argument("--component", type=str, default="vision.block_output")
    parser.add_argument("--n_epochs", type=int, default=5000)
    parser.add_argument("--scheduler_type", choices=["reduce_on_plateau", "cosine", "step"], default=None)
    parser.add_argument("--data_root", type=str, default="datasets/probes")
    parser.add_argument("--modality", choices=["vision", "connector", "language"], default="vision")
    parser.add_argument("--task", choices=["position"], default="position")
    parser.add_argument("--chart_type", choices=["bar", "line"], default="bar")
    parser.add_argument("--segmentation_units", type=str, default=None, nargs="+")
    parser.add_argument("--generalization", choices=["cs", "pos", "task"], default="cs")
    parser.add_argument("--binary", action="store_true")
    parser.add_argument("--save_probes", action="store_true", default=False)
    args = parser.parse_args()

    model_name = args.model_id.split('/')[-1]
    if "fugu" in args.model_id.lower():  # Finetuned model
        model_name = f"fugu-{model_name}"

    if -1 in args.layer:  # TODO: make this work for all models (only works for Llama-3.2-11B-Vision-Instruct)
        if "vision" in args.component:
            if "llama" in args.model_id.lower():
                args.layer = [3, 7, 15, 23, 30, 31]
            elif "internvl" in args.model_id.lower():
                args.layer = [23]
            elif "llava" in args.model_id.lower():
                args.layer = [25]
        elif "language" in args.component:
            if "llama" in args.model_id.lower():
                args.layer = list(range(40))
            elif "internvl" in args.model_id.lower():
                args.layer = [l for l in list(range(47))]
            elif "llava" in args.model_id.lower():
                args.layer = [l for l in list(range(28))]

    if "language" in args.component:
        args.segmentation_units = None

    if args.segmentation_units is None or None in args.segmentation_units or "None" in args.segmentation_units:
        args.segmentation_units = None

    base_colors = ["red", "blue", "green", "orange"]
    additional_colors = ["purple", "brown", "cyan", "magenta"]

    if args.chart_type == "bar":
        target_attributes = base_colors + additional_colors
        base_output_dir = os.path.join(args.data_root, "bar")

        for target_color in target_attributes:
            dataset_root = os.path.join(base_output_dir, target_color)
            if not _dataset_exists(dataset_root):
                create_bar_probe_dataset(
                    target_color=target_color,
                    target_values=range(1, 9),
                    n_points=args.n_points,
                    base_output_dir=base_output_dir,
                )

            for layer in args.layer:
                probe = train_probe(
                    model_id=model_name if "fugu" in model_name else args.model_id,
                    modality=args.modality,
                    task=args.task,
                    generalization=args.generalization,
                    target_color=target_color,
                    target_shape="bar",
                    chart_type="bar",
                    layer=layer,
                    component=args.component,
                    data_root=dataset_root,
                    segmentation_units=args.segmentation_units,
                    n_epochs=args.n_epochs,
                    scheduler_type=args.scheduler_type,
                )

                if args.save_probes and probe is not None:
                    os.makedirs("probes", exist_ok=True)
                    probe_path = f"probes/PROBE_{model_name}_{args.component.replace('.', '-').replace('_', '-')}_{args.task}_{target_color}_layer{layer}"
                    torch.save(probe, probe_path)

    else:  # line charts
        target_attributes = list(range(1, 9))
        base_output_dir = os.path.join(args.data_root, "line")

        for target_x in target_attributes:
            dataset_root = os.path.join(base_output_dir, f"x_{target_x}")
            if not _dataset_exists(dataset_root):
                create_line_probe_dataset(
                    target_x=target_x,
                    target_y_values=range(1, 9),
                    n_points=args.n_points,
                    base_output_dir=base_output_dir,
                )

            for layer in args.layer:
                probe = train_probe(
                    model_id=model_name if "fugu" in model_name else args.model_id,
                    modality=args.modality,
                    task=args.task,
                    generalization=args.generalization,
                    target_color="blue",
                    target_shape="line",
                    chart_type="line",
                    target_x_value=target_x,
                    layer=layer,
                    component=args.component,
                    data_root=dataset_root,
                    segmentation_units=args.segmentation_units,
                    n_epochs=args.n_epochs,
                    scheduler_type=args.scheduler_type,
                )

                if args.save_probes and probe is not None:
                    os.makedirs("probes", exist_ok=True)
                    probe_path = f"probes/PROBE_{model_name}_{args.component.replace('.', '-').replace('_', '-')}_{args.task}_x{target_x}_layer{layer}"
                    torch.save(probe, probe_path)