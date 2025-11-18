import os
import glob
import argparse
from collections import Counter
from itertools import chain
import math
import random
import copy
import uuid
from typing import List, Dict, Tuple, Optional, Union
import json
import tqdm
import ast
import cv2

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DynamicCache

from models.models import get_model_handler
from utils.generate_prompts import generate_prompt, SYMBOL_TO_SHAPE_MAP, QUERIES, PROMPT_PREFIX, PROMPT_PREFIX_CONCISE
from analyze_results import validate_extraction_with_llm, get_task_answer
from utils.stimuli import StimulusPlotter, StimulusConfig, generate_segmentation_for_metadata

from pyvene import (  # Use the local pyvene version! 
    IntervenableModel,
    VanillaIntervention,
    IntervenableConfig,
)

from transformers import logging as transformers_logging
transformers_logging.set_verbosity_error()



class CounterfactualDataset(Dataset):
    """A PyTorch Dataset class for handling counterfactual image pairs and their metadata.

    This dataset class is designed to work with pairs of images where one image is a counterfactual
    variation of another. It handles loading and processing of images, metadata, and segmentation
    masks for both the original and counterfactual images.

    Attributes:
        pairs (pd.DataFrame): DataFrame containing pairs of image IDs (source_id and base_id)
        task_type (str): Type of task being performed (e.g., "count", "position", "distance")
        segmentation_units (List[str]): List of segmentation units to consider
        root_dir (str): Root directory containing the data
        vit_patch_size (int): Size of ViT patches for segmentation
        image_resolution (int): Resolution of the input images
    """
    def __init__(
        self,
        pairs: pd.DataFrame,
        task_type: str,
        segmentation_units: List[str],
        root_dir: str = "datasets/intervention_data",
        vit_patch_size: int = 14,
        image_resolution: Union[int, Tuple[int, int]] = 560,
        multi_image_crops: bool = False,
    ):
        self.pairs = pairs
        self.task_type = task_type
        self.segmentation_units = segmentation_units
        self.root_dir = root_dir
        self.vit_patch_size = vit_patch_size
        self.image_resolution = image_resolution

        self.ids = pairs['source_id'].unique().tolist() + pairs['base_id'].unique().tolist()

        self.metadatas = {}
        self.images = {}
        count = 0
        for i in tqdm.tqdm(self.ids, desc="Loading counterfactual segmentations"):
            self.metadatas[i] = np.load(f"{self.root_dir}/metadata/{i}.npy", allow_pickle=True).item()
            self.images[i] = Image.open(f"{self.root_dir}/stimuli/{i}.png")
            self.metadatas[i]['segmentation_path'] = self.metadatas[i]['image_path'].replace('/stimuli/', '/segmentation/').replace('.png', '.npy')

            if self.metadatas[i]['segmentation_path'] is None or not os.path.exists(self.metadatas[i]['segmentation_path']): 
                self.metadatas[i]['segmentation_path'] = self.metadatas[i]['image_path'].replace('/stimuli/', '/segmentation/').replace('.png', '.npy')
                generate_segmentation_for_metadata(self.metadatas[i]['metadata_path'])
            
            if f'segmentation_patchified_{image_resolution}x{vit_patch_size}' not in self.metadatas[i]:
                seg = np.load(self.metadatas[i]['segmentation_path'], allow_pickle=True).item()
                final_seg = self._process_segmentation(seg)
                if multi_image_crops:
                    final_seg = [final_seg]

                    seg_crops = []
                    for si in range(2):
                        for sj in range(2):
                            seg_crop = {}
                            for sk, sv in seg.items():
                                seg_list = []
                                for sl in sv:
                                    resized_seg = cv2.resize(sl, (image_resolution*2, image_resolution*2), interpolation=cv2.INTER_NEAREST)
                                    seg_list.append(resized_seg[si*image_resolution:(si+1)*image_resolution, sj*image_resolution:(sj+1)*image_resolution])
                                seg_crop[sk] = seg_list
                            seg_crops.append(seg_crop)
                    
                    for seg_crop in seg_crops:
                        final_seg.append(self._process_segmentation(seg_crop))

                self.metadatas[i][f'segmentation_patchified_{image_resolution}x{vit_patch_size}'] = final_seg
                #np.save(self.metadatas[i]['metadata_path'], self.metadatas[i], allow_pickle=True)
        
        new_pairs = []
        for i, row in self.pairs.iterrows():
            row_copy = row.copy()
            seg = self.get_cf_segmentation(row)
            if seg is None:
                continue
            row_copy["segmentation"] = str(seg)
            row_copy["base_image_path"] = self.metadatas[row["base_id"]]["image_path"]
            row_copy["source_image_path"] = self.metadatas[row["source_id"]]["image_path"]

            base_metadata = self.metadatas[row["base_id"]]
            source_metadata = self.metadatas[row["source_id"]]
            
            if task_type == "position":
                # Find the point in base that's not in source
                base_points = np.array(base_metadata['points'])
                source_points = np.array(source_metadata['points'])
                
                # Find the index of the point that's different
                if len(base_points) == 1:
                    diff_idx = 0
                else:
                    diff_idx = None
                    for i, point in enumerate(base_points):
                        if not any(np.array_equal(point, p) for p in source_points):
                            diff_idx = i
                            break
                
                row_copy["base_answer"] = str(base_metadata['points'][diff_idx])
                row_copy["source_answer"] = str(source_metadata['points'][diff_idx])
                
                # Get the color and shape of the target object
                target_color = base_metadata['plot_color'][diff_idx]
                target_shape = SYMBOL_TO_SHAPE_MAP[base_metadata['plot_dot_shape'][diff_idx]]

                # Generate prompts
                base_prompt = f"What is the (x, y) coordinate of the {target_color} {target_shape}? Please format your response as an ordered pair of values enclosed within parentheses."
                base_answer_template = f"The (x, y) coordinate of the {target_color} {target_shape} is ("
                
                # Update the prompts in the pairs DataFrame
                row_copy["base_prompt"] = base_prompt
                row_copy["source_prompt"] = base_prompt
                row_copy["base_answer_template"] = base_answer_template
                row_copy["source_answer_template"] = base_answer_template

                new_pairs.append(row_copy)

            elif task_type == "mean":
                # Extract objects from base and source
                base_points = np.array(base_metadata['points'])
                source_points = np.array(source_metadata['points'])

                # Calculate mean of base and source points
                base_mean = np.round(np.mean(base_points, axis=0)).astype(int)
                source_mean = np.round(np.mean(source_points, axis=0)).astype(int)
                
                row_copy["base_answer"] = str(base_mean)
                row_copy["source_answer"] = str(source_mean)

                base_prompt = "What is the (x, y) coordinate that is closest to the centroid, or arithmetic mean of the positions of all data points? Please round both the x-value and y-value to the nearest whole number."
                base_answer_template = "The (x, y) coordinate that is closest to the centroid (arithmetic mean of all data points) is ("
                
                # Update the prompts in the pairs DataFrame
                row_copy["base_prompt"] = base_prompt
                row_copy["source_prompt"] = base_prompt
                row_copy["base_answer_template"] = base_answer_template
                row_copy["source_answer_template"] = base_answer_template

                new_pairs.append(row_copy)
            
            elif task_type == "count":
                # Extract objects from base and source
                base_points = np.array(base_metadata['points'])
                source_points = np.array(source_metadata['points'])

                row_copy["base_answer"] = str(len(base_points))
                row_copy["source_answer"] = str(len(source_points))

                base_prompt = "How many data points are there in this plot?"
                base_answer_template = "The number of data points in the plot is "

                # Update the prompts in the pairs DataFrame
                row_copy["base_prompt"] = base_prompt
                row_copy["source_prompt"] = base_prompt
                row_copy["base_answer_template"] = base_answer_template
                row_copy["source_answer_template"] = base_answer_template

                new_pairs.append(row_copy)
            
            elif task_type == "distance":
                # Extract objects from base prompt
                base_prompt = row_copy["base_prompt"]
                
                # Find all color-shape pairs in the prompts
                base_objects = []
                
                # For base prompt
                for color in base_metadata['plot_color']:
                    for shape in [SYMBOL_TO_SHAPE_MAP[s] for s in base_metadata['plot_dot_shape']]:
                        if f"{color} {shape}" in base_prompt:
                            base_objects.append((color, shape))
                            break

                # Find the point that differs between base and source
                base_points = np.array(base_metadata['points'])
                source_points = np.array(source_metadata['points'])
                
                # Find the index of the point that's different
                diff_idx = None
                for i, point in enumerate(base_points):
                    if not any(np.array_equal(point, p) for p in source_points):
                        diff_idx = i
                        break
                
                # Get the color and shape of the differing point
                diff_color = base_metadata['plot_color'][diff_idx]
                diff_shape = SYMBOL_TO_SHAPE_MAP[base_metadata['plot_dot_shape'][diff_idx]]
                
                # Append to base_objects
                base_objects.append((diff_color, diff_shape))
                
                # Create answer templates
                base_answer_template = QUERIES["distance"]["answer_template"].format(color1=base_objects[0][0], shape1=base_objects[0][1], color2=base_objects[1][0], shape2=base_objects[1][1])
                
                # Update the prompts in the pairs DataFrame
                row_copy["base_answer_template"] = base_answer_template
                row_copy["source_answer_template"] = base_answer_template
                row_copy["source_prompt"] = base_prompt
                
                new_pairs.append(row_copy)
            else:
                if "min" in task_type or "max" in task_type:
                    base_answer_template = QUERIES[task_type.split("_")[0]]["answer_template"].format(axis=task_type.split("_")[1])
                else:
                    base_answer_template = QUERIES[task_type]["answer_template"]
                
                # Update the prompts in the pairs DataFrame
                row_copy["base_answer_template"] = base_answer_template
                row_copy["source_answer_template"] = base_answer_template

                # Get the target axis (x or y) and operation (min or max)
                axis = task_type.split("_")[1]
                operation = task_type.split("_")[0]
                
                # Get coordinates for the target axis
                base_coords = np.array([p[0 if axis == "x" else 1] for p in base_metadata['points']])
                source_coords = np.array([p[0 if axis == "x" else 1] for p in source_metadata['points']])
                
                # Find indices of points at extreme values
                if operation == "min":
                    base_extreme_idx = np.where(base_coords == np.min(base_coords))[0]
                    source_extreme_idx = np.where(source_coords == np.min(source_coords))[0]
                else:  # max
                    base_extreme_idx = np.where(base_coords == np.max(base_coords))[0]
                    source_extreme_idx = np.where(source_coords == np.max(source_coords))[0]
                
                # Get colors and shapes for all extreme points
                base_colors = [base_metadata['plot_color'][i] for i in base_extreme_idx]
                base_shapes = [SYMBOL_TO_SHAPE_MAP[base_metadata['plot_dot_shape'][i]] for i in base_extreme_idx]
                source_colors = [source_metadata['plot_color'][i] for i in source_extreme_idx]
                source_shapes = [SYMBOL_TO_SHAPE_MAP[source_metadata['plot_dot_shape'][i]] for i in source_extreme_idx]
                
                # Format answers as lists of (color, shape) tuples
                row_copy["base_answer"] = str([(c, s) for c, s in zip(base_colors, base_shapes)])
                row_copy["source_answer"] = str([(c, s) for c, s in zip(source_colors, source_shapes)])

                new_pairs.append(row_copy)
        
        self.pairs = pd.DataFrame(new_pairs)
        
    def _patchify_segmentation(self, seg):
        """
        Patchify a segmentation mask into a list of ViT patches.
        """
        if isinstance(self.image_resolution, int):
            self.image_resolution = (self.image_resolution, self.image_resolution)
    
        num_patches_i = self.image_resolution[0] // self.vit_patch_size
        num_patches_j = self.image_resolution[1] // self.vit_patch_size

        # Resize seg to image_resolution
        if seg.shape != self.image_resolution:
            seg = cv2.resize(seg, self.image_resolution, interpolation=cv2.INTER_NEAREST)
        
        if seg.shape[0] % self.vit_patch_size != 0:
            remove_pixels = seg.shape[0] % self.vit_patch_size
            seg = seg[:-remove_pixels, :-remove_pixels]
        
        patches = []
        for i in range(num_patches_i):
            for j in range(num_patches_j):
                patch = seg[i*self.vit_patch_size:(i+1)*self.vit_patch_size, j*self.vit_patch_size:(j+1)*self.vit_patch_size]
                patches.append(patch)

        idxs = []
        for idx, patch in enumerate(patches):
            # Check if any pixel is not white (255)
            if np.any(patch == 1):
                idxs.append(idx + 1) 

        return idxs

    def _process_segmentation(self, seg):
        """
        Recursively process a nested dictionary until reaching lists of arrays.
        
        Args:
            seg: Nested dictionary containing arrays
            patchify_fn: Function to apply to each array
        
        Returns:
            Dictionary with same structure but processed arrays
        """
        if isinstance(seg, dict):
            return {k: self._process_segmentation(v) for k, v in seg.items()}
        elif isinstance(seg, list):
            return [self._patchify_segmentation(arr) for arr in seg]
        else:
            raise ValueError(f"Unexpected type: {type(seg)}")
        
    def get_cf_segmentation(self, pair):
        """
        Process a pair of segmentation masks into Counterfactual Intervention ViT patches.
        """
        def _get_unit_tokens(segs, unit):
            if isinstance(self.image_resolution, int):
                num_patches_i = self.image_resolution // self.vit_patch_size
                num_patches_j = self.image_resolution // self.vit_patch_size
            else:
                num_patches_i = self.image_resolution[0] // self.vit_patch_size
                num_patches_j = self.image_resolution[1] // self.vit_patch_size
            
            if unit == "full_layer":
                num_tokens = (num_patches_i * num_patches_j) + 1
                unit_tokens = list(range(num_tokens))
            elif unit == "dots":
                # Find the dot differences between the two images
                source_dots = set(chain(*segs[1]['dots']))
                base_dots = set(chain(*segs[0]['dots']))
                unit_tokens = list(set(source_dots).symmetric_difference(set(base_dots)))
            elif unit == "cls":
                unit_tokens = [0]
            elif unit == "bg":
                # Get all non-background tokens for source image
                num_tokens = (num_patches_i * num_patches_j) + 1
                source_non_bg = set()
                for seg_type in ['dots', 'axis_x', 'axis_y', 'cls', 'ticks_x', 'ticks_y', 'tick_labels_x', 'tick_labels_y']:
                    if seg_type in segs[1]:
                        source_non_bg.update(chain(*segs[1][seg_type]))
                source_bg = set(range(num_tokens)) - source_non_bg

                # Get all non-background tokens for base image
                base_non_bg = set()
                for seg_type in ['dots', 'axis_x', 'axis_y', 'cls', 'ticks_x', 'ticks_y', 'tick_labels_x', 'tick_labels_y']:
                    if seg_type in segs[0]:
                        base_non_bg.update(chain(*segs[0][seg_type]))
                base_bg = set(range(num_tokens)) - base_non_bg

                unit_tokens = list(source_bg.intersection(base_bg))
            else:
                unit_tokens = []
                for axis in ["x", "y"]:
                    source_axis = set(chain(*segs[1][f"{unit}_{axis}"]))
                    base_axis = set(chain(*segs[0][f"{unit}_{axis}"]))
                    unit_tokens.append(list(set(source_axis).union(set(base_axis))))

            return unit_tokens

        # Process both segmentations
        segs = []
        for i in [pair['base_id'], pair['source_id']]:
            if isinstance(self.image_resolution, int):
                seg = self.metadatas[i][f'segmentation_patchified_{self.image_resolution}x{self.vit_patch_size}']
            else:
                seg = self.metadatas[i][f'segmentation_patchified_{self.image_resolution[0]}x{self.vit_patch_size}']
            segs.append(seg)

        if isinstance(segs[0], list):
            multi_image_crops = True
            num_image_crops = len(segs[0])
        else:
            multi_image_crops = False
        
        cf_segmentation = []

        if not multi_image_crops:
            for unit in self.segmentation_units:
                unit_tokens = _get_unit_tokens(segs, unit)
                cf_segmentation.extend(unit_tokens)
        else:
            for crop_idx in range(num_image_crops):
                try:
                    crop_segs = [segs[0][crop_idx], segs[1][crop_idx]]
                    crop_cf_segmentation = []
                    for unit in self.segmentation_units:
                        unit_tokens = _get_unit_tokens(crop_segs, unit)
                        crop_cf_segmentation.extend(unit_tokens)
                    cf_segmentation.append(crop_cf_segmentation)
                except KeyError:
                    return None
            assert len(cf_segmentation) == num_image_crops

            reshaped_indices = []
            for crop_indices in cf_segmentation:
                if crop_indices:  # Skip empty sublists
                    reshaped_indices.append([crop_indices])  # Add batch dimension of 1
                else:
                    reshaped_indices.append([[]]) 
            cf_segmentation = reshaped_indices

        return cf_segmentation

    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx: int) -> Dict:
        return self.pairs.iloc[idx].to_dict()

def create_counterfactual_stimulus(metadata: Dict, task_type: str, output_dir: str, stimuli_dir: str) -> Dict:
    """
    Create a counterfactual stimulus for a given task and save it.
    
    Args:
        metadata: Original metadata dictionary
        task_type: Task type to generate counterfactual for
        output_dir: Directory to save the counterfactual image and metadata
        stimuli_dir: Base directory for stimuli
        
    Returns:
        Dictionary with paths to the generated counterfactual
    """
    # Create a deep copy of the metadata
    new_metadata = copy.deepcopy(metadata)
    
    # Get current points, colors, and shapes
    points = np.array(metadata['points'])
    colors = metadata['plot_color'] if isinstance(metadata['plot_color'], list) else [metadata['plot_color']]
    shapes = metadata['plot_dot_shape'] if isinstance(metadata['plot_dot_shape'], list) else [metadata['plot_dot_shape']]
    n_points = metadata['n_points']

    # Generate unique ID for the counterfactual
    cf_id = str(uuid.uuid4())
    
    # Generate counterfactual based on task type
    if task_type == "count":
        # For count, add or remove a dot
        add_dot = n_points < 8  # Add up to 8, remove for larger n
        
        if add_dot:
            # Adding a dot: need to find a position that doesn't overlap with existing dots
            all_positions = [(x, y) for x in range(1, 9) for y in range(1, 9)]
            existing_positions = [tuple(p) for p in points]
            available_positions = [p for p in all_positions if p not in existing_positions]
            
            if not available_positions:
                raise ValueError("No available positions to add a new dot")
            
            # Choose a random available position
            new_pos = random.choice(available_positions)
            
            # Choose a random color and shape that doesn't match existing ones
            all_colors = ["red", "blue", "green", "orange"]
            all_shapes = ["o", "^", "*", "s"]
            
            # Try to avoid duplicating colors and shapes if possible
            available_colors = [c for c in all_colors if c not in colors]
            available_shapes = [s for s in all_shapes if s not in shapes]
            
            new_color = random.choice(available_colors if available_colors else all_colors)
            new_shape = random.choice(available_shapes if available_shapes else all_shapes)
            
            # Update metadata
            new_metadata['points'] = np.vstack([points, new_pos]).tolist()
            new_metadata['plot_color'] = colors + [new_color]
            new_metadata['plot_dot_shape'] = shapes + [new_shape]
            new_metadata['n_points'] = len(new_metadata['points'])
            
        else:
            # Removing a dot
            if len(points) <= 1:
                raise ValueError("Cannot remove dots from a stimulus with only one dot")
            
            # Choose a random dot to remove
            idx_to_remove = random.randint(0, len(points) - 1)
            
            # Update metadata
            new_points = np.delete(points, idx_to_remove, axis=0)
            new_metadata['points'] = new_points.tolist()
            new_metadata['plot_color'] = [c for i, c in enumerate(colors) if i != idx_to_remove]
            new_metadata['plot_dot_shape'] = [s for i, s in enumerate(shapes) if i != idx_to_remove]
            new_metadata['n_points'] = len(new_metadata['points'])
            
        new_metadata['id'] = f"{task_type}_BASE_{metadata['id']}_CF_{cf_id}"
        
    elif task_type == "position":
        # For position, move one dot to a new position
        # Select dot to move (for n=1, there's only one dot)
        dot_idx = 0 if n_points == 1 else random.randint(0, n_points - 1)
        
        # Get dimensionality constraint
        dimensionality = metadata.get('dimensionality', '2d')
        
        # Find a new position based on dimensionality
        current_pos = points[dot_idx]
        all_positions = []
        
        if dimensionality == 'x_collinear' and n_points != 1:
            # Keep x-coordinate fixed, change y-coordinate
            fixed_x = current_pos[0]
            all_positions = [(fixed_x, y) for y in range(1, 9) if y != current_pos[1]]
        elif dimensionality == 'y_collinear' and n_points != 1:
            # Keep y-coordinate fixed, change x-coordinate
            fixed_y = current_pos[1]
            all_positions = [(x, fixed_y) for x in range(1, 9) if x != current_pos[0]]
        else:  # 2d
            # Can move to any position in the grid
            all_positions = [(x, y) for x in range(1, 9) for y in range(1, 9)]
            all_positions.remove(tuple(current_pos))
        
        # Filter out positions already occupied by other dots
        existing_positions = [tuple(p) for i, p in enumerate(points) if i != dot_idx]
        available_positions = [p for p in all_positions if p not in existing_positions]
        
        if not available_positions:
            raise ValueError(f"No available positions to move dot with dimensionality {dimensionality}")
        
        # Choose a random available position
        new_pos = random.choice(available_positions)
        
        # Update the point position
        new_points = points.copy()
        new_points[dot_idx] = new_pos
        new_metadata['points'] = new_points.tolist()
        new_metadata['id'] = f"{task_type}_BASE_{metadata['id']}_CF_{cf_id}"
        
    elif task_type == "distance":
        # For distance, move one of the dots to change the distance
        if n_points < 2:
            raise ValueError("Distance task requires at least 2 dots")
        
        # For simplicity, if n > 2, we'll select two dots and move one
        if n_points == 2:
            # Select one dot to move
            move_dot_idx = random.randint(0, 1)
            fixed_dot_idx = 1 - move_dot_idx
        else:
            # For n > 2, select two dots and move one
            dot_indices = random.sample(range(n_points), 2)
            fixed_dot_idx, move_dot_idx = dot_indices
        
        fixed_pos = points[fixed_dot_idx]
        current_pos = points[move_dot_idx]
        
        # Calculate current distance
        current_distance = np.sqrt(np.sum((fixed_pos - current_pos)**2))
        
        # Get dimensionality constraint
        dimensionality = metadata.get('dimensionality', '2d')
        
        # Find a new position based on dimensionality that changes the distance
        all_positions = []
        
        if dimensionality == 'x_collinear':
            # Keep x-coordinate fixed, change y-coordinate
            fixed_x = current_pos[0]
            all_positions = [(fixed_x, y) for y in range(1, 9) if y != current_pos[1]]
        elif dimensionality == 'y_collinear':
            # Keep y-coordinate fixed, change x-coordinate
            fixed_y = current_pos[1]
            all_positions = [(x, fixed_y) for x in range(1, 9) if x != current_pos[0]]
        else:  # 2d
            # Can move to any position in the grid
            all_positions = [(x, y) for x in range(1, 9) for y in range(1, 9)]
            all_positions.remove(tuple(current_pos))
        
        # Filter out positions already occupied by other dots
        existing_positions = [tuple(p) for i, p in enumerate(points) if i != move_dot_idx]
        available_positions = [p for p in all_positions if p not in existing_positions]
        
        if not available_positions:
            raise ValueError(f"No available positions to move dot with dimensionality {dimensionality}")
        
        # Calculate distances for all potential new positions
        distances = [np.sqrt(np.sum((np.array(pos) - fixed_pos)**2)) for pos in available_positions]
        
        # Find positions that result in a different whole-number distance
        current_whole_dist = int(round(current_distance))
        different_dist_indices = [i for i, d in enumerate(distances) if int(round(d)) != current_whole_dist]
        
        if not different_dist_indices:
            raise ValueError("No positions available that would change the whole number distance")
        
        # Choose a random position that changes the distance
        new_pos_idx = random.choice(different_dist_indices)
        new_pos = available_positions[new_pos_idx]
        
        # Update the point position
        new_points = points.copy()
        new_points[move_dot_idx] = new_pos
        new_metadata['points'] = new_points.tolist()
        new_metadata['id'] = f"{task_type}_BASE_{metadata['id']}_CF_{cf_id}"
        
    elif task_type in ["min_x", "min_y", "max_x", "max_y"]:
        # For min/max tasks, move all dots at the extreme value
        if n_points < 2:
            raise ValueError(f"{task_type} task requires at least 2 dots")
        
        # Determine axis and whether it's min or max
        axis = 0 if task_type.endswith('x') else 1
        is_min = task_type.startswith('min')
        
        # Find all dots at the current extreme value
        if is_min:
            current_extreme_val = np.min(points[:, axis])
            extreme_indices = np.where(points[:, axis] == current_extreme_val)[0]
        else:
            current_extreme_val = np.max(points[:, axis])
            extreme_indices = np.where(points[:, axis] == current_extreme_val)[0]
        
        # Get dimensionality constraint
        dimensionality = metadata.get('dimensionality', '2d')
        
        # Generate possible new positions based on dimensionality
        if dimensionality == 'x_collinear' and axis == 0:
            return None
        elif dimensionality == 'x_collinear' and axis == 1:
            # For x_collinear with y axis, keep y fixed but can change x
            possible_positions = [(points[extreme_indices[0]][0], y) for y in range(1, 9)]
            possible_positions = [p for p in possible_positions if p not in [tuple(points[i]) for i in extreme_indices]]
        elif dimensionality == 'y_collinear' and axis == 0:
            # For y_collinear with x axis, keep x fixed but can change y
            possible_positions = [(x, points[extreme_indices[0]][1]) for x in range(1, 9)]
            possible_positions = [p for p in possible_positions if p not in [tuple(points[i]) for i in extreme_indices]]
        elif dimensionality == 'y_collinear' and axis == 1:
            return None
        else:
            # For 2d or mismatched dimensionality, can move anywhere
            possible_positions = [(x, y) for x in range(1, 9) for y in range(1, 9)]
            possible_positions = [p for p in possible_positions if p not in [tuple(points[i]) for i in extreme_indices]]
        
        # Filter out positions occupied by other dots
        existing_positions = [tuple(p) for i, p in enumerate(points) if i not in extreme_indices]
        available_positions = [p for p in possible_positions if p not in existing_positions]
        
        if not available_positions:
            return None
        
        # Find positions that would make a different dot the new min/max
        if is_min:
            # For min, find positions that make another dot the new min
            # First find the second smallest value
            sorted_values = np.sort(points[:, axis])
            second_smallest = sorted_values[len(extreme_indices)] if len(sorted_values) > len(extreme_indices) else None
            
            if second_smallest is None:
                return None
                
            # Find positions that would make these dots' values greater than the second smallest
            valid_positions = [p for p in available_positions if p[axis] > second_smallest]
        else:
            # For max, find positions that make another dot the new max
            # First find the second largest value
            sorted_values = np.sort(points[:, axis])
            second_largest = sorted_values[-(len(extreme_indices)+1)] if len(sorted_values) > len(extreme_indices) else None
            
            if second_largest is None:
                return None
                
            # Find positions that would make these dots' values less than the second largest
            valid_positions = [p for p in available_positions if p[axis] < second_largest]
        
        if not valid_positions:
            return None
        
        # Choose random valid positions for each extreme dot
        new_points = points.copy()
        for idx in extreme_indices:
            if not valid_positions:
                break
            new_pos = random.choice(valid_positions)
            valid_positions.remove(new_pos)
            new_points[idx] = new_pos
        
        new_metadata['points'] = new_points.tolist()
        new_metadata['id'] = f"{task_type}_BASE_{metadata['id']}_CF_{cf_id}"
        
    elif task_type == "mean":
        # For mean task, shift the entire cluster while preserving relative positions
        # Calculate current mean
        current_mean = np.mean(points, axis=0)
        
        # Determine a random shift that keeps all points within the grid
        min_vals = np.min(points, axis=0)
        max_vals = np.max(points, axis=0)
        
        # Range for shifting: can't move below 1 or above 8
        min_shift = np.array([1, 1]) - min_vals
        max_shift = np.array([8, 8]) - max_vals
        
        # Calculate the possible range for shifts
        x_shifts = list(range(int(min_shift[0]), int(max_shift[0]) + 1))
        y_shifts = list(range(int(min_shift[1]), int(max_shift[1]) + 1))
        
        # Remove 0 to ensure we actually move the cluster
        if 0 in x_shifts:
            x_shifts.remove(0)
        if 0 in y_shifts:
            y_shifts.remove(0)
        
        # If we can't shift in either dimension, return None
        if not x_shifts or not y_shifts:
            raise ValueError("Cannot shift the cluster while keeping all points within grid")
        
        # Choose random shifts
        shift_x = random.choice(x_shifts)
        shift_y = random.choice(y_shifts)
        shift = np.array([shift_x, shift_y])
        
        # Apply the shift to all points
        new_points = points + shift
        new_metadata['points'] = new_points.tolist()
        new_metadata['id'] = f"{task_type}_BASE_{metadata['id']}_CF_{cf_id}"
    else:
        raise ValueError(f"Unknown task type: {task_type}")
    
    # Update stats
    if new_metadata['axis_config'] == "xy":
        points_array = np.array(new_metadata['points'])
        new_metadata.update({
            "mean": np.mean(points_array, axis=0).tolist() if len(points_array) > 0 else [0, 0],
            "median": np.median(points_array, axis=0).tolist() if len(points_array) > 0 else [0, 0],
            "std": np.std(points_array, axis=0).tolist() if len(points_array) > 0 else [0, 0],
            "min": np.min(points_array, axis=0).tolist() if len(points_array) > 0 else [0, 0],
            "max": np.max(points_array, axis=0).tolist() if len(points_array) > 0 else [0, 0]
        })
    else:
        points_array = np.array(new_metadata['points']).flatten()
        new_metadata.update({
            "mean": float(np.mean(points_array)) if len(points_array) > 0 else 0,
            "median": float(np.median(points_array)) if len(points_array) > 0 else 0,
            "std": float(np.std(points_array)) if len(points_array) > 0 else 0,
            "min": float(np.min(points_array)) if len(points_array) > 0 else 0,
            "max": float(np.max(points_array)) if len(points_array) > 0 else 0
        })
    
    # Create directories if they don't exist
    os.makedirs(os.path.join(output_dir, "metadata"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "stimuli"), exist_ok=True)
    
    # Save metadata
    cf_metadata_path = os.path.join(output_dir, "metadata", f"{new_metadata['id']}.npy")
    new_metadata['metadata_path'] = cf_metadata_path
    
    # Save image path
    cf_image_path = os.path.join(output_dir, "stimuli", f"{new_metadata['id']}.png")
    new_metadata['image_path'] = cf_image_path
    
    # Recreate the config
    config = StimulusConfig(
        n_points=new_metadata['n_points'],
        axis_config=new_metadata['axis_config'],
        color=new_metadata['plot_color'],
        dot_size=new_metadata['plot_dot_size'],
        dot_shape=new_metadata['plot_dot_shape'],
        x_offset=new_metadata.get('x_offset', 0),
        y_offset=new_metadata.get('y_offset', 0)
    )
    
    # Create image
    plotter = StimulusPlotter()
    image = plotter.plot_stimulus_without_segmentation(
        points=np.array(new_metadata['points']),
        config=config,
        x_offset=config.x_offset,
        y_offset=config.y_offset
    )
    
    # Save image
    image.save(cf_image_path)
    
    # Save metadata
    np.save(cf_metadata_path, new_metadata, allow_pickle=True)

    # Append task type to metadata
    new_metadata['task_type'] = task_type
    
    # Generate prompt and answer for the counterfactual
    prompt, answer_template, expected_answer = generate_prompt(new_metadata)
    new_metadata['expected_answer'] = expected_answer
    np.save(cf_metadata_path, new_metadata, allow_pickle=True)
    
    # Return information about the counterfactual
    return {
        'id': new_metadata['id'],
        'original_id': metadata['id'],
        'metadata_path': cf_metadata_path,
        'image_path': cf_image_path,
        'task_type': task_type,
        'prompt': prompt,
        'answer_template': answer_template,
        'expected_answer': expected_answer
    }


def test_model_on_counterfactual(model, metadata: Dict, task_type: str, use_cot: bool = False) -> Tuple[bool, str, str]:
    """
    Test if a model correctly answers a prompt for a given counterfactual stimulus using an LLM judge.
    
    Args:
        model: Model handler object
        metadata: Metadata for the stimulus
        task_type: Task type (count, position, etc.)
        use_cot: Whether to use chain-of-thought prompting
        
    Returns:
        Tuple of (is_correct, model_response, expected_answer)
    """
    # Load image
    image = Image.open(metadata['image_path'])

    # Append task type to metadata
    metadata['task_type'] = task_type
    
    # Generate prompt
    prompt, answer_template, expected_answer = generate_prompt(metadata)
    
    # Add prefix
    full_prompt = f"{PROMPT_PREFIX}\n{prompt}" if use_cot else f"{PROMPT_PREFIX_CONCISE}\n{prompt}"
    
    # Process inputs
    inputs = model.process_input(image, full_prompt, template=answer_template if not use_cot else None)
    
    # Get model prediction
    model_response = model.generate(**inputs)
    
    # Use the LLM judge to evaluate correctness
    
    # Convert metadata to a format compatible with get_task_answer
    task_row = {
        'task_type': task_type,
        'points': json.dumps(metadata['points']),
        'color': json.dumps(metadata['plot_color']),
        'dot_shape': json.dumps(metadata['plot_dot_shape']),
        'n_points': metadata['n_points'],
        'answer': json.dumps(expected_answer)
    }
    
    # Get the expected answer in the format needed for validation
    if task_type == "mean":
        # For mean task, compute arithmetic mean from points
        points = np.array(metadata['points'])
        mean_x = np.mean(points[:, 0])
        mean_y = np.mean(points[:, 1])
        answer = [mean_x, mean_y]
        answer = [
            (math.floor(answer[0]), math.floor(answer[1])),
            (math.floor(answer[0]), math.ceil(answer[1])),
            (math.ceil(answer[0]), math.floor(answer[1])),
            (math.ceil(answer[0]), math.ceil(answer[1]))
        ]
    else:
        answer = get_task_answer(task_row)
    
    # Validate the model's response
    validation_result = validate_extraction_with_llm(
        task_type=task_type,
        question=prompt,
        answer=answer,
        full_response=model_response
    )
    
    # Extract the correctness from the validation result
    is_correct = validation_result['correct']
    
    return is_correct, model_response, str(expected_answer)


def get_counterfactual_dataset(
    model_id,
    task_type,
    stimuli_dir,
    behavior_dir="results",
    save_dir="datasets/intervention_data",
    device="cuda" if torch.cuda.is_available() else "cpu",
    max_pairs_per_n=100,
):
    """
    Generates a counterfactual dataset for a given model and task type.
    
    Args:
        model_id: Model identifier
        model_behavioral_results: DataFrame with model's behavioral results
        task_type: Task type to generate counterfactuals for
        stimuli_dir: Directory containing stimuli files
        save_dir: Directory to save counterfactual data
        device: Device to run model on
        max_pairs_per_n: Maximum number of counterfactual pairs to generate per n_points value
        
    Returns:
        DataFrame containing counterfactual pairs
    """
    # Create output directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(f"{save_dir}/metadata", exist_ok=True)
    os.makedirs(f"{save_dir}/stimuli", exist_ok=True)
    
    # Try to load existing counterfactual dataset
    model_name = model_id.split('/')[-1]
    
    output_file = f"{save_dir}/datasets/{model_name}/{task_type}.csv"
    os.makedirs(f"{save_dir}/datasets/{model_name}", exist_ok=True)
    
    try:
        dataset = pd.read_csv(output_file)
        print(f"Loaded existing counterfactual dataset with {len(dataset)} pairs")
        return dataset
    except FileNotFoundError:
        print("No existing counterfactual dataset found. Creating a new one...")

    # Load model behavioral results
    model_results_files = glob.glob(f"{behavior_dir}/{model_name}/fugu/behavioral_analysis/*.json")
    
    if not model_results_files:
        raise ValueError(f"No behavioral results found for model {model_id}")
    
    # Load all behavioral results
    model_behavioral_results = []
    for result_file in model_results_files:
        with open(result_file, 'r', encoding='utf-8') as f:
            result = json.load(f)
            model_behavioral_results.append(result)
    
    model_behavioral_results = pd.DataFrame(model_behavioral_results)

    if "templated" not in model_behavioral_results["generation_type"].unique():
        result_type = "none"
    else:
        result_type = "templated"
    
    # Filter behavioral results to only include specific query, templated generation, & correct predictions
    filtered_results = model_behavioral_results[
        (model_behavioral_results["task_type"] == task_type) &
        (model_behavioral_results["generation_type"] == result_type) &
        (model_behavioral_results["validation_correct"] == True)
    ]

    if task_type == "mean":
        filtered_results = filtered_results[filtered_results["point_generator"] == "mean_centered"]
    
    if len(filtered_results) == 0:
        raise ValueError(f"No stimuli found for {model_id} with task_type={task_type}, generation_type=templated, and validation_correct.")
    
    print(f"Loaded {len(filtered_results)} stimuli with task_type={task_type}, generation_type=templated, and validation_correct.")
    
    # Load model
    model = get_model_handler(model_id, device=device)
    
    # Group by n_points to ensure we have pairs for each n value
    grouped_results = filtered_results.groupby('n_points')
    
    # Check for existing counterfactual images to avoid duplication
    existing_cf_files = glob.glob(f"{save_dir}/stimuli/*.png")
    existing_cf_ids = [os.path.basename(f).replace('.png', '') for f in existing_cf_files]
    
    counterfactual_pairs = []
    
    for n_points, group in grouped_results:
        print(f"Processing stimuli with n_points = {n_points}...")
        
        # Only process if we have correct results
        if len(group) == 0:
            print(f"No correct results found for n_points = {n_points}")
            continue

        if task_type == "count" and n_points >= 8:
            print(f"Skipping n_points = {n_points} for count task")
            continue

        if task_type in ["min_x", "min_y", "max_x", "max_y", "mean"] and n_points <= 2:
            print(f"Skipping n_points = {n_points} for {task_type} task")
            continue
            
        # Track counterfactual pairs for this n_points value
        pairs_for_n = []
        
        # First try to use existing counterfactual images
        for _, row in tqdm.tqdm(group.iterrows(), desc=f"Checking existing counterfactuals for n={n_points}", total=len(group)):
            # Skip if we already have enough pairs
            if len(pairs_for_n) >= max_pairs_per_n:
                break
                
            # Load metadata for original stimulus
            original_metadata = np.load(f"{stimuli_dir}/metadata/{row['id']}.npy", allow_pickle=True).item()
            
            # Check for existing counterfactuals for this stimulus
            potential_cf_ids = [cf_id for cf_id in existing_cf_ids if cf_id.startswith(f"{task_type}_BASE_{row['id']}_CF_")]
            
            for cf_id in potential_cf_ids:
                # Load counterfactual metadata
                cf_metadata_path = f"{save_dir}/metadata/{cf_id}.npy"
                
                if not os.path.exists(cf_metadata_path):
                    continue
                    
                cf_metadata = np.load(cf_metadata_path, allow_pickle=True).item()
                
                # Check if stimulus id exists in behavioral results
                is_correct = cf_metadata['id'] in model_behavioral_results['id'].values
                if not is_correct:
                    # If not found in behavioral results, test the model
                    is_correct, _, _ = test_model_on_counterfactual(model, cf_metadata, task_type, use_cot=False)
                
                if is_correct:
                    # Both original and counterfactual answers are correct
                    prompt, _, expected_answer = generate_prompt(cf_metadata)
                    
                    pairs_for_n.append({
                        'base_id': original_metadata['id'],
                        'source_id': cf_metadata['id'],
                        'base_prompt': row['prompt'],
                        'source_prompt': prompt,
                        'base_answer': row['answer'],
                        'source_answer': str(expected_answer)
                    })
                    
                    if len(pairs_for_n) >= max_pairs_per_n:
                        break
        
        # If we haven't collected enough pairs, generate new ones
        attempts = 0
        while len(pairs_for_n) < max_pairs_per_n and attempts < 5:
            print(f"Generated {len(pairs_for_n)} pairs from existing counterfactuals. "
                  f"Generating {max_pairs_per_n - len(pairs_for_n)} more...")
            
            # Shuffle the correct results to get a random subset
            correct_results_shuffled = group.sample(frac=1).reset_index(drop=True)
            
            for _, row in tqdm.tqdm(correct_results_shuffled.iterrows(), 
                                  desc=f"Generating new counterfactuals for n={n_points}", 
                                  total=len(correct_results_shuffled)):
                # Skip if we already have enough pairs
                if len(pairs_for_n) >= max_pairs_per_n:
                    break
                
                # Load metadata for original stimulus
                original_metadata = np.load(f"{stimuli_dir}/metadata/{row['id']}.npy", allow_pickle=True).item()
                
                # Try to generate a counterfactual
                #try:
                # Create the counterfactual
                cf_info = create_counterfactual_stimulus(
                    original_metadata, 
                    task_type, 
                    save_dir, 
                    stimuli_dir
                )
                if cf_info is None:
                    continue
                
                # Load the counterfactual metadata
                cf_metadata = np.load(cf_info['metadata_path'], allow_pickle=True).item()
                
                # Test model on the counterfactual
                is_correct, _, _ = test_model_on_counterfactual(model, cf_metadata, task_type, use_cot=False)
                
                if is_correct:
                    # Both original and counterfactual answers are correct
                    pairs_for_n.append({
                        'base_id': original_metadata['id'],
                        'source_id': cf_metadata['id'],
                        'base_prompt': row['prompt'],
                        'source_prompt': cf_info['prompt'],
                        'base_answer': row['answer'],
                        'source_answer': str(cf_info['expected_answer'])
                    })
                        
                #except Exception as e:
                    #print(f"Error generating counterfactual: {e}")
                    #continue
            
            attempts += 1
            if attempts >= 5:
                print(f"Reached maximum attempts ({attempts}) for generating counterfactuals. "
                      f"Proceeding with {len(pairs_for_n)} pairs.")
                break
        
        print(f"Collected {len(pairs_for_n)} pairs for n_points = {n_points}")
        counterfactual_pairs.extend(pairs_for_n)
    
    # Create DataFrame from counterfactual pairs
    df = pd.DataFrame(counterfactual_pairs)
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"Saved {len(df)} counterfactual pairs to {output_file}")
    
    return df


def vanilla_config(
    model_type,
    component,
    layer,
    full_layer_swap=False,
    intervention_type=VanillaIntervention,
    num_crops=1
):
    # Set up num_patches interventions
    representations = []
    if full_layer_swap:
        representations.extend(
            [
                {
                    "layer": layer,
                    "component": component,
                    "intervention_type": intervention_type,
                }
            ] 
        )
    else:
        representations.extend(
            [
                {
                    "layer": layer,
                    "component": component,
                    "unit": "pos",
                    #"max_number_of_units": 1,
                    "intervention_type": intervention_type,
                }
            ] * num_crops
        )

    config = IntervenableConfig(
        model_type=model_type,
        representations=representations,
        # Intervene on base at the same time
        mode="parallel",
    )
    return config


def detect_repetition(tokens, window_sizes=[4, 5, 6], threshold=3):
    """
    Check for repetitive patterns in generated tokens.
    
    Args:
        tokens: List of token ids
        window_sizes: List of pattern lengths to check
        threshold: Number of repetitions before stopping
        
    Returns:
        bool: True if repetitive pattern detected
    """
    if len(tokens) < min(window_sizes) * 2:
        return False

    for n in window_sizes:
        matches = []
        for idx, _ in enumerate(tokens):
            tmp_string = tokens[idx:idx+n]
            matches.append(str(tmp_string))
        for key, val in Counter(matches).items():
            if val >= threshold:
                return True
            
    return False


def run_intervention(
    intervenable,
    dataloader,
    layer,
    task_type,
    full_layer_swap=False,
    is_llama=False,
    is_llava=False,
    cls_token=False,
    segmentation_units=None,
):
    results = []

    if is_llava:
        try:
            with open(f"llava_{task_type}_cf_idx.txt", 'r') as file:
                lines = file.readlines()
                cf_idx = [int(line.strip()) for line in lines]
            new_run = False
        except FileNotFoundError:
            cf_idx = []
            new_run = True

    # Iterate through batches
    for idx, pair in tqdm.tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Running intervention (layer={layer}; task_type={task_type}; segmentation_units={segmentation_units})..."):
        segmentation = json.loads(pair["segmentation"][0])
        if len(segmentation) == 0:
            continue

        # Unfold any lists in segmentation while preserving integers
        if full_layer_swap:
            segmentation = None
        elif not is_llava:
            cls_token = True  # TODO: change this
            segmentation = [item for x in segmentation for item in (x if isinstance(x, list) else [x])]
            segmentation = list(set(segmentation))
            segmentation = [s - (not cls_token) for s in segmentation]
            segmentation = [s for s in segmentation if s >= 0]
        else:
            for crop_idx in range(len(segmentation)):
                for batch_idx in range(len(segmentation[crop_idx])):
                    seg = segmentation[crop_idx][batch_idx]
                    seg = [item for x in seg for item in (x if isinstance(x, list) else [x])]
                    seg = list(set(seg))
                    seg = [s - (not cls_token) for s in seg]
                    seg = [s for s in seg if s >= 0]
                    segmentation[crop_idx][batch_idx] = seg
            segmentation = (segmentation, segmentation)

        source_inputs = intervenable.process_input(
            Image.open(pair["source_image_path"][0]),
            f"{PROMPT_PREFIX_CONCISE}\n{pair['source_prompt'][0]}",
            pair["source_answer_template"][0]
        )
        base_inputs = intervenable.process_input(
            Image.open(pair["base_image_path"][0]),
            f"{PROMPT_PREFIX_CONCISE}\n{pair['base_prompt'][0]}",
            pair["base_answer_template"][0]
        )

        if task_type == "position":
            pair["base_answer"][0] = json.loads(pair["base_answer"][0])
            pair["source_answer"][0] = json.loads(pair["source_answer"][0])
        elif task_type == "mean":
            base_answer = pair["base_answer"][0].replace("[", "").replace("]", "").split(" ")
            base_answer = [int(base_answer[0]), int(base_answer[1])]
            pair["base_answer"][0] = base_answer
            source_answer = pair["source_answer"][0].replace("[", "").replace("]", "").split(" ")
            source_answer = [int(source_answer[0]), int(source_answer[1])]
            pair["source_answer"][0] = source_answer
        else:
            try:
                pair["base_answer"][0] = ast.literal_eval(pair["base_answer"][0])
                pair["source_answer"][0] = ast.literal_eval(pair["source_answer"][0])
            except SyntaxError:
                print(task_type)
                print(pair["base_answer"][0], type(pair["base_answer"][0]))
                print(pair["source_answer"][0], type(pair["source_answer"][0]))
                print(1 / 0)
        
        if full_layer_swap:
            unit_locations = None
        else:
            #cls_token = True  # TODO: change this
            unit_locations = {
                #"sources->base": [s - (not cls_token) for s in segmentation],  # TODO: change this
                "sources->base": segmentation,
            }

        if is_llava:
            if new_run and idx not in cf_idx:
                base_out = intervenable.model.generate(**base_inputs)
                source_out = intervenable.model.generate(**source_inputs)

                base_out_text = intervenable.processor.decode(base_out[0], skip_special_tokens=True)
                source_out_text = intervenable.processor.decode(source_out[0], skip_special_tokens=True)
                stub = pair["base_answer_template"][0]

                if task_type == "position" or task_type == "mean":
                    base_out_answer = base_out_text.split(stub)[-1].split("(")[-1].split(")")[0].split(", ")
                    base_out_answer = [int(base_out_answer[0]), int(base_out_answer[1])]
                    source_out_answer = source_out_text.split(stub)[-1].split("(")[-1].split(")")[0].split(", ")
                    source_out_answer = [int(source_out_answer[0]), int(source_out_answer[1])]
                    if not (base_out_answer == pair["base_answer"][0] and source_out_answer == pair["source_answer"][0]):
                        continue
                    else:
                        cf_idx.append(idx)
                elif task_type == "distance":
                    base_out_answer = base_out_text.split(stub)[-1].split(".")[0].strip()
                    base_out_answer = int(base_out_answer)
                    source_out_answer = source_out_text.split(stub)[-1].split(".")[0].strip()
                    source_out_answer = int(source_out_answer)
                    if not (base_out_answer in pair["base_answer"][0] and source_out_answer in pair["source_answer"][0]):
                        continue
                    else:
                        cf_idx.append(idx)
                elif task_type == "count":
                    base_out_answer = base_out_text.split(stub)[-1].split(".")[0].strip()
                    base_out_answer = int(base_out_answer)
                    source_out_answer = source_out_text.split(stub)[-1].split(".")[0].strip()
                    source_out_answer = int(source_out_answer)
                    if not (base_out_answer == pair["base_answer"][0] and source_out_answer == pair["source_answer"][0]):
                        continue
                    else:
                        cf_idx.append(idx)
                else:
                    base_out_answer = base_out_text.split(stub)[-1].replace(".", "").strip().split(" ")
                    base_out_answer = [attribute for attribute in base_out_answer if len(attribute) > 0]
                    base_out_answer = tuple(base_out_answer)
                    source_out_answer = source_out_text.split(stub)[-1].replace(".", "").strip().split(" ")
                    source_out_answer = [attribute for attribute in source_out_answer if len(attribute) > 0]
                    source_out_answer = tuple(source_out_answer)
                    if not (base_out_answer in pair["base_answer"][0] and source_out_answer in pair["source_answer"][0]):
                        continue
                    else:
                        cf_idx.append(idx)
            elif idx not in cf_idx:
                continue

        # Initialize variables for iterative generation
        current_length = base_inputs["input_ids"].shape[1]
        max_length = current_length + 100
        generated_tokens = []
        past_key_values = None
        
        if is_llama:
            while current_length < max_length:
                # Run intervention
                _, counterfactual_outputs = intervenable(
                    base={**base_inputs, "use_cache": True, "past_key_values": past_key_values},
                    sources={**source_inputs, "use_cache": True, "past_key_values": past_key_values},
                    unit_locations=unit_locations,
                    use_cache=True,
                )
                
                # Get next token
                next_token = counterfactual_outputs.logits[:, -1, :].argmax(dim=-1)
                past_key_values = counterfactual_outputs.past_key_values  # Save cache for next iteration
                generated_tokens.append(next_token.item())
                
                # Update input_ids, attention_mask, and cross_attention_mask for both base and source
                base_inputs = {
                    "input_ids": next_token.unsqueeze(0),  # [1, 1]
                    "attention_mask": torch.ones(1, current_length + 1, device=base_inputs["attention_mask"].device),
                    # Adjust cross attention mask to match expected dimensions
                    "use_cache": True,
                    "past_key_values": past_key_values,
                    "cross_attention_mask": base_inputs["cross_attention_mask"][:, :1, :, :]
                }
                #if is_llama:
                #    base_inputs["cross_attention_mask"] = base_inputs["cross_attention_mask"][:, :1, :, :]  # Take only first timestep

                source_inputs = {
                    "input_ids": next_token.unsqueeze(0),
                    "attention_mask": torch.ones(1, current_length + 1, device=source_inputs["attention_mask"].device),
                    "use_cache": True,
                    "past_key_values": past_key_values,
                    "cross_attention_mask": source_inputs["cross_attention_mask"][:, :1, :, :]
                }
                #if is_llama:
                #    source_inputs["cross_attention_mask"] = source_inputs["cross_attention_mask"][:, :1, :, :]  # Take only first timestep
                
                current_length += 1
                
                # Check for EOS token
                if hasattr(intervenable.processor, 'eos_token') and next_token.item() == intervenable.processor.eos_token:
                    break
                if detect_repetition(generated_tokens):
                    break
            
            # Decode the full generated sequence
            output_text = intervenable.processor.decode(generated_tokens, skip_special_tokens=True)
        else:
            while current_length < max_length:
                _, counterfactual_outputs = intervenable(
                    base=base_inputs,
                    sources=source_inputs,
                    unit_locations=unit_locations,
                    use_cache=False,
                )
                next_token = counterfactual_outputs.logits[:, -1, :].argmax(dim=-1)
                generated_tokens.append(next_token.item())
                current_length += 1

                if hasattr(intervenable.processor, 'eos_token') and next_token.item() == intervenable.processor.eos_token:
                    break
                if detect_repetition(generated_tokens):
                    break
                if "llava" in intervenable.model.config.architectures[0].lower() and intervenable.processor.decode(next_token) == "assistant":
                    break

                if "llava" in intervenable.model.config.architectures[0].lower():
                    base_inputs = {
                        "pixel_values": base_inputs["pixel_values"],
                        "input_ids": torch.cat([base_inputs["input_ids"], next_token.unsqueeze(0)], dim=1),  # [1, 1]
                        "attention_mask": torch.ones(1, current_length + 1, device=base_inputs["attention_mask"].device),
                        "image_sizes": base_inputs["image_sizes"],
                        "batch_num_images": base_inputs["batch_num_images"],
                    }

                    source_inputs = {
                        "pixel_values": source_inputs["pixel_values"],
                        "input_ids": torch.cat([source_inputs["input_ids"], next_token.unsqueeze(0)], dim=1),  # [1, 1]
                        "attention_mask": torch.ones(1, current_length + 1, device=source_inputs["attention_mask"].device),
                        "image_sizes": source_inputs["image_sizes"],
                        "batch_num_images": source_inputs["batch_num_images"],
                    }
                else:
                    base_inputs = {
                        "pixel_values": base_inputs["pixel_values"],
                        "input_ids": torch.cat([base_inputs["input_ids"], next_token.unsqueeze(0)], dim=1),  # [1, 1]
                        "attention_mask": torch.ones(1, current_length + 1, device=base_inputs["attention_mask"].device),
                    }

                    source_inputs = {
                        "pixel_values": source_inputs["pixel_values"],
                        "input_ids": torch.cat([source_inputs["input_ids"], next_token.unsqueeze(0)], dim=1),  # [1, 1]
                        "attention_mask": torch.ones(1, current_length + 1, device=source_inputs["attention_mask"].device),
                    }

                torch.cuda.empty_cache()

            output_text = intervenable.processor.decode(generated_tokens, skip_special_tokens=True)

        results.append(
            {
                **pair,
                "counterfactual_full_response": output_text,
            }
        )

    if is_llava and new_run:
        with open(f"llava_{task_type}_cf_idx.txt", 'w') as file:
            for idx in cf_idx:
                file.write(f"{idx}\n")

    results = pd.DataFrame(results)
    
    return results


def get_vanilla_intervention_results(
    model_id,
    task_type,
    segmentation_units,
    layers,
    component,
    stimuli_dir="datasets/intervention_data",
    behavior_dir="results",
    save_dir="results/vanilla_interventions",
    device="cuda" if torch.cuda.is_available() else "cpu",
    result_id="",
    subsample_size=-1,
):
    """
    Get vanilla intervention results for a given query
    
    Args:
        model_id: Model identifier
        task_type: Task type to generate counterfactuals for
        stimuli_dir: Directory containing stimuli files
        behavior_dir: Directory containing behavioral results
        save_dir: Directory to save intervention results
        device: Device to run model on
    
    Returns:
        DataFrame containing intervention results
    """
    # Create output directory if it doesn't exist
    model_name = model_id.split('/')[-1]
    os.makedirs(f"{save_dir}/{model_name}", exist_ok=True)

    behavior_dir = f"{behavior_dir}/{model_name}/fugu"
    
    # Get counterfactual dataset
    counterfactual_dataset = get_counterfactual_dataset(
        model_id=model_id,
        task_type=task_type,
        behavior_dir=behavior_dir,
        stimuli_dir=stimuli_dir,
        device=device,
    )
    
    if len(counterfactual_dataset) == 0:
        raise ValueError(f"No counterfactual pairs generated for {model_id} with {task_type} task type.")
    print(f"Loaded {len(counterfactual_dataset)} counterfactual pairs.")

    # Load model
    model = get_model_handler(model_id, device=device)
    
    if layers == "all":
        layers = model.get_layers(modality=component.split(".")[0])

    full_layer_swap = segmentation_units == ["full_layer"]

    counterfactual_dataset = CounterfactualDataset(
        counterfactual_dataset,
        task_type,
        segmentation_units,
        image_resolution=model.get_image_resolution(),
        multi_image_crops=("llava" in model_id.lower() and not full_layer_swap),
    )
    counterfactual_dataloader = DataLoader(counterfactual_dataset, batch_size=1, shuffle=False)

    if subsample_size > 0:  # Randomly subsample the counterfactual dataset
        indices = list(range(len(counterfactual_dataloader)))
        indices = indices[:subsample_size]
        subset_dataset = torch.utils.data.Subset(counterfactual_dataset, indices)
        counterfactual_dataloader = DataLoader(subset_dataset, batch_size=1, shuffle=False)
        subsample_str = f"_subsample{subsample_size}"
    else:
        subsample_str = ""

    basic_results_config = {
        "model_id": model_id,
        "task_type": task_type,
        "segmentation_units": segmentation_units,
        "component": component,
        "modality": component.split(".")[0],
        "full_layer_swap": full_layer_swap,
        "stimuli_dir": stimuli_dir,
        "behavior_dir": behavior_dir,
        "subsample_size": subsample_size,
    }

    full_results = []

    model_type = type(model.model)
    wrapped_model = model.model

    if "llava" in model_id.lower() and not full_layer_swap:
        num_crops = 5
    else:
        num_crops = 1

    # Iterate over model layers
    for layer in layers:
        # Create the intervention with two swaps: one content swap and one null swap
        intervenable_config = vanilla_config(
            model_type,  # Model type, e.g. LlamaForConditionalGeneration
            component,  # Which model component to intervene on, e.g. "vision.block_output"
            layer,  # Which layer to intervene on, e.g. 0
            full_layer_swap=full_layer_swap,  # Are we swapping the full layer?
            intervention_type=VanillaIntervention,  # Intervention type
            num_crops=num_crops,
        )
        # Set up the intervenable model
        intervenable = IntervenableModel(intervenable_config, wrapped_model, is_llava="llava" in model_id.lower())
        intervenable.set_device(model.device)
        intervenable.disable_model_gradients()
        intervenable.processor = model.processor
        intervenable.process_input = model.process_input

        results = run_intervention(
            intervenable,
            counterfactual_dataloader,
            layer,
            task_type,
            full_layer_swap=full_layer_swap,
            is_llama="llama" in model_id.lower(),
            is_llava="llava" in model_id.lower(),
            cls_token=model.cls_token,
            segmentation_units=segmentation_units,
        )
        
        for k, v in basic_results_config.items():
            results[k] = [v] * len(results)
        results["layer"] = [layer] * len(results)

        results.to_csv(f"{save_dir}/{model_name}/{result_id}_layer{layer}{subsample_str}.csv", index=False)

        full_results.append(results)

    return pd.concat(full_results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, required=True, 
                       help="HuggingFace model ID",
                       choices=["meta-llama/Llama-3.2-11B-Vision-Instruct",
                                "Salesforce/blip-vqa-base",
                                "allenai/Molmo-7B-D-0924",
                                "allenai/Molmo-7B-O-0924",
                                "facebook/chameleon-7b",
                                "OpenGVLab/InternVL3-14B-hf",
                                "llava-hf/llava-onevision-qwen2-7b-ov-hf",
                                "fugu/Llama-3.2-11B-Vision-Instruct"])
    parser.add_argument("--task_type", type=str, 
                       default="position", 
                       choices=["count", "position", "distance", "min_x", "min_y", "max_x", "max_y", "mean"],
                       help="Task type to generate counterfactuals for")
    parser.add_argument("--segmentation_units", type=str, nargs="+", default=None)
    parser.add_argument("--layers", type=int, nargs="+", default=None)
    parser.add_argument("--component", type=str, default="vision.block_input")
    parser.add_argument("--stimuli_dir", type=str,
                       default="datasets/intervention_data",
                       help="Directory containing stimuli and metadata")
    parser.add_argument("--save_dir", type=str,
                       default="results/vanilla_interventions",
                       help="Directory to save results")
    parser.add_argument("--behavior_dir", type=str, 
                       default="results",
                       help="Directory containing behavioral results")
    parser.add_argument("--device", type=str, 
                       default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device to run model on")
    parser.add_argument("--subsample_size", type=int, default=-1, help="Counterfactual dataset subsample size to run intervention on per layer")
    
    args = parser.parse_args()

    if args.segmentation_units is None:
        args.segmentation_units = ["full_layer"]  # Swap full layer

    if args.layers is None:
        args.layers = "all"
    
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    result_id = str(uuid.uuid4())
    
    results = get_vanilla_intervention_results(
        model_id=args.model_id,
        task_type=args.task_type,
        segmentation_units=args.segmentation_units,
        layers=args.layers,
        component=args.component,
        stimuli_dir=args.stimuli_dir,
        behavior_dir=args.behavior_dir,
        save_dir=args.save_dir,
        device=args.device,
        result_id=result_id,
        subsample_size=args.subsample_size,
    )

    model_name = args.model_id.split('/')[-1]
    results.to_csv(f"{args.save_dir}/{model_name}/{result_id}.csv", index=False)


if __name__ == "__main__":
    main()