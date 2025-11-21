import os
import shutil
from typing import Optional, Union, List, Tuple, Dict
from abc import ABC, abstractmethod
from dataclasses import dataclass
import itertools
import string
import uuid
import tqdm
import yaml

import numpy as np
from PIL import Image

import matplotlib.pyplot as plt


def highlight_tokens(
    image_path: Union[str, Image.Image],
    token_indices: Union[int, List[int], List[List[int]]],
    patch_size: int = 14,
    highlight_colors: List[Tuple[int, int, int]] = [(0, 0, 0)],
    alpha: float = 0.5,
    cls_token: bool = True
) -> Image.Image:
    """Create a transparent mask highlighting specified token patches with different colors.
    Mostly used for debugging segmentation masks.
    
    Args:
        image_path: Path to image or PIL Image object
        token_indices: Token indices to highlight. Can be:
            - Single integer
            - List of integers
            - List of lists of integers (for multiple color groups)
        patch_size: Size of patches in pixels
        highlight_colors: List of RGB color tuples for highlight masks
        alpha: Transparency of highlight mask (0-1)
        cls_token: Whether to remove cls token from token indices
    
    Returns:
        PIL Image with highlighted tokens
    """
    # Normalize token indices format
    if not any(isinstance(x, list) for x in token_indices) and len(highlight_colors) > 1:
        if len(token_indices) != len(highlight_colors):
            raise ValueError("Number of indices must match number of colors when using multiple colors")
        token_indices = [[idx] for idx in token_indices]
    
    if isinstance(token_indices, int):
        token_indices = [[token_indices]]
    elif isinstance(token_indices[0], int):
        token_indices = [token_indices]
        
    if len(token_indices) > len(highlight_colors):
        highlight_colors = highlight_colors * len(token_indices)
    
    # Load and process image
    im = Image.open(image_path).convert('RGB') if isinstance(image_path, str) else image_path.convert('RGB')
    img_array = np.array(im)
    h, w = img_array.shape[:2]
    n_h, n_w = h // patch_size, w // patch_size
    
    # Create and fill highlight mask
    mask = np.zeros((h, w, 4), dtype=np.uint8)
    for indices, color in zip(token_indices, highlight_colors):
        for idx in indices:
            if cls_token:
                idx -= 1
            i, j = idx // n_w, idx % n_w  # Convert to 2D coordinates
            mask[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size] = [*color, int(255 * alpha)]
    
    # Composite mask with original image
    mask = Image.fromarray(mask, mode='RGBA')
    im = im.convert('RGBA')
    return Image.alpha_composite(im, mask)


def tokenize_segmentations(
    segmentations: Dict[str, np.ndarray],
    patch_size: int,
    percent_content_threshold: float = 0.015
) -> Dict[str, List[int]]:
    """
    Convert segmentation masks into token indices for a vision transformer encoder.
    
    This function divides each segmentation mask into patches of size patch_size x patch_size
    and identifies which patches contain meaningful content (non-white pixels) above a given
    threshold. The indices of these patches are returned as tokens, with +1 offset to account
    for the class token.
    
    Args:
        segmentations: Dictionary mapping segmentation types to their corresponding masks
        patch_size: Size of each patch in pixels (both height and width)
        percent_content_threshold: Minimum percentage of non-white pixels required for a patch
                                 to be considered meaningful (default: 0.015)
    
    Returns:
        Dictionary mapping segmentation types to lists of token indices
    """
    tokenized_segmentations = {}

    for seg_type, segmentation in segmentations.items():
        type_segmentations = []

        for seg in segmentation:
            token_idxs = []
            h, w = seg.shape
            n_h = h // patch_size
            n_w = w // patch_size

            for j in range(n_h):
                for k in range(n_w):
                    idx = j * n_w + k
                    patch = seg[
                        j * patch_size:(j + 1) * patch_size,
                        k * patch_size:(k + 1) * patch_size
                    ]

                    if seg_type == 'bg':
                        if np.all(patch == 1):
                            token_idxs.append(idx + 1)
                    else:
                        has_content = np.any(patch == 1)
                        percent_content = np.sum(patch == 1) / patch_size**2
                            
                        if has_content and percent_content > percent_content_threshold:
                            token_idxs.append(idx + 1)  # +1 for cls token

            type_segmentations.append(token_idxs)

        tokenized_segmentations[seg_type] = type_segmentations
        
    return tokenized_segmentations


def generate_segmentation_for_metadata(metadata_file: str) -> None:
    """Generate segmentation masks for a stimulus given its metadata file"""
    # Load metadata
    metadata = np.load(metadata_file, allow_pickle=True).item()
    
    # Reconstruct config from metadata
    config = StimulusConfig(
        n_points=metadata['n_points'],
        axis_config=metadata['axis_config'],
        color=metadata['plot_color'],
        dot_size=metadata['plot_dot_size'],
        dot_shape=metadata['plot_dot_shape'],
        x_offset=metadata['x_offset'],
        y_offset=metadata['y_offset'],
        tick_scale=metadata.get('tick_scale', 8 + 1),
        n_ticks=metadata.get('n_ticks', 8),
        axis_range=metadata.get('axis_range', (0, 8))
    )
    
    # Create plotter
    plotter = StimulusPlotter()
    
    # Generate segmentation masks
    # The new metadata schema stores `grid_points` (absolute grid co-ordinates) and
    # `points` (relative to tick labels).  For backward-compatibility we fall back
    # to the old `points` field if `grid_points` is missing.
    if 'grid_points' in metadata:
        points = np.array(metadata['grid_points'])
    else:
        points = np.array(metadata['points'])

    _, seg_dict = plotter.plot_stimulus(
        points=points,
        config=config,
        x_offset=config.x_offset,
        y_offset=config.y_offset,
        n_ticks=config.n_ticks,
        tick_scale=config.tick_scale
    )
    
    # Save segmentation masks
    """  TODO: FIX
    segmentation_path = metadata['segmentation_path']
    if segmentation_path is None:
        # Generate new segmentation path if none exists
        segmentation_path = metadata['image_path'].replace('/stimuli/', '/segmentation/').replace('.png', '.npy')
    """
    segmentation_path = metadata['image_path'].replace('/stimuli/', '/segmentation/').replace('.png', '.npy')
    os.makedirs(os.path.dirname(segmentation_path), exist_ok=True)
    np.save(segmentation_path, seg_dict, allow_pickle=True)
    
    # Update metadata with segmentation path if it was None
    if metadata['segmentation_path'] is None:
        metadata['segmentation_path'] = segmentation_path
        np.save(metadata['metadata_path'], metadata, allow_pickle=True)


@dataclass
class StimulusConfig:
    """Configuration for a single stimulus"""
    n_points: int
    axis_config: str  # 'x', 'y', or 'xy'
    color: Union[str, List[str]]  # e.g. "red", "blue", "green"
    dot_size: Union[int, List[int]]  # e.g. 100, 500
    dot_shape: Union[str, List[str]]  # e.g. "o", "s", "^"
    chart_type: str = "scatter"  # scatter, bar, or line
    axis_label: Optional[Union[str, List[str]]] = None  # e.g. "EVALUATE:f'{var}'.upper()/VAR:axis_config"
    axis_range: Tuple[int, int] = (0, 8)
    tick_scale: int = 8 + 1
    n_ticks: int = 8
    tick_labels: Optional[List[Union[str, int]]] = None
    x_tick_labels: Optional[List[Union[str, int]]] = None
    y_tick_labels: Optional[List[Union[str, int]]] = None
    dot_tick_offset: float = 0
    dot_axis_offset: float = 0.225
    x_offset: float = 0  # Horizontal offset (-1 to 1)
    y_offset: float = 0  # Vertical offset (-1 to 1)
    spatial_config: Optional[str] = None

    def __post_init__(self):
        # For color
        if self.chart_type not in ['bar', 'line']:
            if isinstance(self.color, str):
                self.plot_color = [self.color] * self.n_points
            else:
                # If it's a list, use it exactly as provided
                self.plot_color = self.color
        else:
            # Colors will be assigned in StimulusGenerator based on chart rules
            self.plot_color = []
        
        # For dot size
        if isinstance(self.dot_size, int):
            self.plot_dot_size = [self.dot_size] * self.n_points
        else:
            # If it's a list, use it exactly as provided
            self.plot_dot_size = self.dot_size
        
        # For dot shape
        if isinstance(self.dot_shape, str):
            self.plot_dot_shape = [self.dot_shape] * self.n_points
        else:
            # If it's a list, use it exactly as provided
            self.plot_dot_shape = self.dot_shape

class PointGenerator(ABC):
    """Abstract base class for point generation strategies"""
    
    @abstractmethod
    def generate_points(self, config: StimulusConfig) -> np.ndarray:
        """Generate points based on the configuration"""
        pass

    @abstractmethod
    def sample_points(self, config: StimulusConfig, n_samples: int) -> List[np.ndarray]:
        """Sample n unique sets of points"""
        pass

    @abstractmethod
    def get_metadata(self) -> Dict:
        """Return metadata about the generator's parameters"""
        pass

class FixedPointGenerator(PointGenerator):
    """Returns coordinates of a fixed position"""

    def __init__(self, x_position: List[int], y_position: List[int]):
        self.x_position = x_position
        self.y_position = y_position

    def generate_points(self, config: StimulusConfig) -> np.ndarray:
        if len(self.x_position) != config.n_points or len(self.y_position) != config.n_points:
            raise ValueError(f"Length of x_position ({len(self.x_position)}) and y_position ({len(self.y_position)}) must match n_points ({config.n_points})")
            
        if config.axis_config == 'xy':
            return np.array([[x, y] for x, y in zip(self.x_position, self.y_position)])
        elif config.axis_config == 'x':
            return np.array([[x] for x in self.x_position])
        elif config.axis_config == 'y':
            return np.array([[y] for y in self.y_position])
    
    def sample_points(self, config: StimulusConfig, n_samples: int = 1) -> List[np.ndarray]:
        # For grid points, each sample is already unique
        return [self.generate_points(config) for _ in range(n_samples)]

    def get_metadata(self) -> Dict:
        return {
            'stimulus_type': 'fixed',
            'x_position': self.x_position,
            'y_position': self.y_position
        }

    def to_string(self):
        return "grid"

class GridPointGenerator(PointGenerator):
    """Generates points in a grid in the specified range"""

    def __init__(self, x_position: int = None, y_position: int = None):
        self.x_position = x_position
        self.y_position = y_position

    def generate_points(self, config: StimulusConfig) -> np.ndarray:
        """Generate points that snap to displayed tick marks rather than the unit grid.

        The visible tick marks are determined by `config.axis_range` and `config.n_ticks`. If
        `axis_range[1] == 8` and `n_ticks == 8` (the original behaviour) the tick positions
        are `[0, 1, 2, 3, 4, 5, 6, 7, 8]` and the previous implementation that sampled from
        `range(1, 9)` is reproduced (we continue to **exclude** the `0` position that lies
        on the axis for aesthetic reasons).

        For other tick configurations we first compute the *grid interval* (distance between
        consecutive tick marks in data-space units) and then build the list of *visible*
        tick positions.  We again exclude the first tick (position `0`) because points are
        intended to appear in the positive quadrant, away from the axes.
        """

        # Distance between consecutive tick marks in grid units
        grid_interval = max(1, config.axis_range[1] // config.n_ticks)

        # Build list of tick positions (exclude 0 – the axis line)
        positions = list(range(grid_interval, config.axis_range[1] + 1, grid_interval))

        if self.x_position is not None:
            x_positions = [self.x_position] * len(positions)
            y_positions = positions
        elif self.y_position is not None:
            x_positions = positions
            y_positions = [self.y_position] * len(positions)
        else:
            x_positions = positions
            y_positions = positions

        if config.axis_config == 'xy':
            # Generate all possible grid positions snapping to tick marks
            grid_positions = list(itertools.product(x_positions, y_positions))
            sampled_indices = np.random.choice(len(grid_positions), config.n_points, replace=False)
            sampled_positions = [grid_positions[i] for i in sampled_indices]
            return np.array(sampled_positions)
        else:
            sampled_indices = np.random.choice(len(positions), config.n_points, replace=False)
            sampled_positions = [positions[i] for i in sampled_indices]
            return np.array(sampled_positions)
    
    def sample_points(self, config: StimulusConfig, n_samples: int = 1) -> List[np.ndarray]:
        # For grid points, each sample is already unique
        return [self.generate_points(config) for _ in range(n_samples)]

    def get_metadata(self) -> Dict:
        return {
            'stimulus_type': 'grid',
            'x_position': self.x_position,
            'y_position': self.y_position
        }

    def to_string(self):
        return "grid"

class UniformPointGenerator(PointGenerator):
    """Generates points uniformly in the specified range"""

    def __init__(self, x_position: int = None, y_position: int = None):
        self.x_position = x_position
        self.y_position = y_position

    def generate_points(self, config: StimulusConfig) -> np.ndarray:
        if config.axis_config == 'xy':
            # Add small buffer to prevent points at exact min/max
            buffer = (config.axis_range[1] - config.axis_range[0]) * 0.05
            pos = np.random.uniform(
                config.axis_range[0] + buffer, 
                config.axis_range[1] - buffer, 
                (config.n_points, 2)
            )
            if self.x_position is not None:
                pos[:, 0] = self.x_position
            if self.y_position is not None:
                pos[:, 1] = self.y_position
            return pos
        else:
            # Add small buffer to prevent points at exact min/max
            buffer = (config.axis_range[1] - config.axis_range[0]) * 0.05
            return np.random.uniform(
                config.axis_range[0] + buffer,
                config.axis_range[1] - buffer,
                config.n_points
            )
    
    def sample_points(self, config: StimulusConfig, n_samples: int = 1) -> List[np.ndarray]:
        # For continuous distributions, each sample is already unique with probability 1
        return [self.generate_points(config) for _ in range(n_samples)]

    def get_metadata(self) -> Dict:
        return {
            'stimulus_type': 'uniform',
            'x_position': self.x_position,
            'y_position': self.y_position
        }

    def to_string(self):
        return "uniform"

class MeanCenteredPointGenerator(PointGenerator):
    """Generates points centered around a mean value"""
    
    def __init__(self, mean: Union[float, Tuple[float, float]], std: float = 1.0):
        self.mean = mean
        self.std = std
    
    def generate_points(self, config: StimulusConfig) -> np.ndarray:
        if config.axis_config == 'xy':
            return np.random.normal(
                self.mean, 
                self.std, 
                (config.n_points, 2)
            )
        else:
            if isinstance(self.mean, tuple) or isinstance(self.mean, list):
                if config.axis_config == 'x':
                    mean = self.mean[0]
                else:
                    mean = self.mean[1]
            else:
                mean = self.mean
            
            return np.random.normal(
                mean,
                self.std,
                config.n_points
            )
    
    def sample_points(self, config: StimulusConfig, n_samples: int = 1) -> List[np.ndarray]:
        # For continuous distributions, each sample is already unique with probability 1
        return [self.generate_points(config) for _ in range(n_samples)]

    def get_metadata(self) -> Dict:
        return {
            'stimulus_type': 'mean_centered',
            'mean': self.mean,
            'std': self.std
        }

    def to_string(self):
        return "mean_centered"

class CorrelationPointGenerator(PointGenerator):
    """Generates points with a specified correlation coefficient and intercept"""
    
    def __init__(self, r: float, intercept: float = 0, noise: float = 0.2):
        self.r = r  # correlation coefficient
        self.intercept = intercept
        self.noise = noise
    
    def get_metadata(self) -> Dict:
        """Return metadata about the correlation parameters"""
        return {
            'stimulus_type': 'correlation',
            'r': self.r,
            'intercept': self.intercept,
            'noise': self.noise
        }
    
    def generate_points(self, config: StimulusConfig) -> np.ndarray:
        n = config.n_points
        points = np.zeros((n, 2))
        valid_points = 0
        axis_range = config.axis_range
        
        while valid_points < n:
            # Generate x values uniformly in the grid range
            x = np.random.uniform(axis_range[0] + 1, axis_range[1])
            
            # Calculate y value to achieve desired correlation
            # Using the formula: y = rx + sqrt(1-r^2)z + intercept
            # where z is random normal noise
            z = np.random.normal(0, 1)
            y = self.r * x + np.sqrt(1 - self.r**2) * z * self.noise + self.intercept
            
            # Only keep points that fall within the grid boundaries
            if axis_range[0] + 1 <= y <= axis_range[1]:
                points[valid_points] = [x, y]
                valid_points += 1
        
        return points
    
    def sample_points(self, config: StimulusConfig, n_samples: int = 1) -> List[np.ndarray]:
        return [self.generate_points(config) for _ in range(n_samples)]

    def to_string(self):
        return f"correlation_r{self.r}_i{self.intercept}_n{self.noise}"

class FunctionPointGenerator(PointGenerator):
    """Generates points following a specified function with noise"""
    
    def __init__(self, func_type: str, noise: float = 0.2, **func_params):
        """
        Args:
            func_type: One of 'linear', 'quadratic', 'exponential', 'log'
            noise: Standard deviation of Gaussian noise
            func_params: Parameters specific to each function type
                linear: slope, intercept
                quadratic: a (curvature), h (x-shift), k (y-shift)
                exponential: a, b (ae^(bx))
                log: a, b (a*log(x) + b)
        """
        self.func_type = func_type
        self.noise = noise
        self.func_params = func_params
        
        # Define function mappings
        self.functions = {
            'linear': lambda x: self.func_params.get('slope', 1) * x + self.func_params.get('intercept', 0),
            'quadratic': lambda x: (
                self.func_params.get('a', 0.2) * 
                (x - self.func_params.get('h', 4))**2 + 
                self.func_params.get('k', 4)
            ),
            'exponential': lambda x: self.func_params.get('a', 1) * np.exp(self.func_params.get('b', 1) * x),
            'log': lambda x: self.func_params.get('a', 1) * np.log(x) + self.func_params.get('b', 0)
        }
    
    def get_metadata(self) -> Dict:
        """Return metadata about the function parameters"""
        return {
            'stimulus_type': 'function',
            'function_type': self.func_type,
            'noise': self.noise,
            'function_params': self.func_params
        }

    def generate_points(self, config: StimulusConfig) -> np.ndarray:
        n = config.n_points
        points = np.zeros((n, 2))
        valid_points = 0
        axis_range = config.axis_range
        
        # For log function, we need x > 0
        x_min = axis_range[0] + 1.1 if self.func_type == 'log' else axis_range[0] + 0.25
        
        while valid_points < n:
            # Generate x value uniformly in the grid range
            if self.func_type == 'quadratic':
                # For quadratic, ensure x values span around the vertex
                h = self.func_params.get('h', 4)
                # Generate points within ±3 units of vertex, but within grid bounds
                x = np.random.uniform(max(axis_range[0] + 0.25, h - 3), min(axis_range[1], h + 3))
            else:
                x = np.random.uniform(x_min, axis_range[1])
            
            # Calculate base y value from function
            base_y = self.functions[self.func_type](x)
            
            # Add noise
            y = base_y + np.random.normal(0, self.noise)
            
            # Only keep points that fall within the grid boundaries
            if axis_range[0] + 0.1 <= y <= axis_range[1]:
                points[valid_points] = [x, y]
                valid_points += 1
        
        return points
    
    def sample_points(self, config: StimulusConfig, n_samples: int = 1) -> List[np.ndarray]:
        return [self.generate_points(config) for _ in range(n_samples)]

    def to_string(self):
        return f"function_{self.func_type}_n{self.noise}"

class ClusterPointGenerator(PointGenerator):
    """Generates points forming two clusters and a query point"""
    
    def __init__(self, cluster_stds: List[float], query_cluster: int = None, ambiguity: float = 0.0):
        """
        Args:
            cluster_stds: Standard deviations for the two clusters
            query_cluster: Which cluster (0 or 1) the query point belongs to. If None, randomly chosen.
            ambiguity: Float between 0 and 1 controlling query point placement:
                      0.0 = at host cluster center (clear membership)
                      1.0 = halfway between host and non-host (maximum ambiguity)
        """
        if not 0 <= ambiguity <= 1:
            raise ValueError("Ambiguity must be between 0 and 1")
        
        self.cluster_stds = cluster_stds
        self.query_cluster = query_cluster
        self.ambiguity = ambiguity
        self.centers = None
        self.actual_query_cluster = None
        self.n_points = None
    
    def generate_points(self, config: StimulusConfig) -> np.ndarray:
        # Calculate points per cluster
        n_per_cluster = (config.n_points - 1) // 2  # -1 for query point
        self.n_points = config.n_points
        axis_range = config.axis_range
        
        # Generate two well-separated cluster centers
        while True:
            center1 = np.random.uniform(axis_range[0] + 1, axis_range[1] - 1, 2)
            center2 = np.random.uniform(axis_range[0] + 1, axis_range[1] - 1, 2)
            if np.linalg.norm(center1 - center2) > 3:  # Ensure good separation
                break
        
        self.centers = np.array([center1, center2])
        
        # Generate first cluster
        cluster1 = np.zeros((n_per_cluster, 2))
        for i in range(n_per_cluster):
            while True:
                point = np.random.normal(center1, self.cluster_stds[0])
                if axis_range[0] + 1 <= point[0] <= axis_range[1] and axis_range[0] + 1 <= point[1] <= axis_range[1]:
                    cluster1[i] = point
                    break
        
        # Generate second cluster
        cluster2 = np.zeros((n_per_cluster, 2))
        for i in range(n_per_cluster):
            while True:
                point = np.random.normal(center2, self.cluster_stds[1])
                if axis_range[0] + 1 <= point[0] <= axis_range[1] and axis_range[0] + 1 <= point[1] <= axis_range[1]:
                    cluster2[i] = point
                    break
        
        # Choose host cluster for query point
        self.actual_query_cluster = np.random.randint(2) if self.query_cluster is None else self.query_cluster
        host_center = self.centers[self.actual_query_cluster]
        other_center = self.centers[1 - self.actual_query_cluster]
        
        # Calculate midpoint between clusters
        midpoint = (host_center + other_center) / 2
        
        # Calculate base position for query point based on ambiguity
        # Interpolate between host center and midpoint
        base_position = host_center + self.ambiguity * (midpoint - host_center)
        
        # Generate query point around the base position
        # Use the host cluster's std for noise, scaled by ambiguity
        query_std = self.cluster_stds[self.actual_query_cluster] * (1 + self.ambiguity)
        
        query_point = None
        while query_point is None:
            # Generate point with noise around base position
            point = np.random.normal(base_position, query_std)
            
            if axis_range[0] + 1 <= point[0] <= axis_range[1] and axis_range[0] + 1 <= point[1] <= axis_range[1]:
                # For high ambiguity, we don't need to check distances
                # For low ambiguity, ensure it's still closer to host
                if self.ambiguity > 0.8 or np.linalg.norm(point - host_center) < np.linalg.norm(point - other_center):
                    query_point = point
        
        # Stack all points in order: cluster1, cluster2, query
        return np.vstack([cluster1, cluster2, query_point.reshape(1, 2)])
    
    def get_metadata(self) -> Dict:
        """Return metadata about the cluster parameters"""
        if self.centers is None:
            raise ValueError("Points have not been generated yet")
        
        return {
            'stimulus_type': 'cluster',
            'cluster_stds': self.cluster_stds,
            'cluster_centers': self.centers.tolist(),
            'query_cluster': self.actual_query_cluster,
            'ambiguity': self.ambiguity,
            'n_points_per_cluster': [(self.n_points - 1) // 2, (self.n_points - 1) // 2]
        }

    def sample_points(self, config: StimulusConfig, n_samples: int = 1) -> List[np.ndarray]:
        return [self.generate_points(config) for _ in range(n_samples)]

    def to_string(self):
        return f"cluster_std{self.cluster_stds[0]}_{self.cluster_stds[1]}_a{self.ambiguity}"

class SingleClusterPointGenerator(PointGenerator):
    """Generates points forming a single cluster (used as base for outlier generation)"""
    
    def __init__(self, cluster_std: float):
        """
        Args:
            cluster_std: Standard deviation for the cluster
        """
        self.cluster_std = cluster_std
        self.center = None
        self.n_points = None
    
    def generate_points(self, config: StimulusConfig) -> np.ndarray:
        # All points belong to one cluster
        self.n_points = config.n_points
        axis_range = config.axis_range
        
        # Generate one cluster center
        while True:
            center = np.random.uniform(axis_range[0] + 1, axis_range[1] - 1, 2)
            self.center = center
            break
        
        # Generate all points in the single cluster
        cluster_points = np.zeros((config.n_points, 2))
        for i in range(config.n_points):
            while True:
                point = np.random.normal(center, self.cluster_std)
                if axis_range[0] + 1 <= point[0] <= axis_range[1] and axis_range[0] + 1 <= point[1] <= axis_range[1]:
                    cluster_points[i] = point
                    break
        
        return cluster_points
    
    def get_metadata(self) -> Dict:
        """Return metadata about the cluster parameters"""
        if self.center is None:
            raise ValueError("Points have not been generated yet")
        
        return {
            'stimulus_type': 'single_cluster',
            'cluster_std': self.cluster_std,
            'cluster_center': self.center.tolist(),
            'n_points': self.n_points
        }

    def sample_points(self, config: StimulusConfig, n_samples: int = 1) -> List[np.ndarray]:
        return [self.generate_points(config) for _ in range(n_samples)]

    def to_string(self):
        return f"single_cluster_std{self.cluster_std}"


class OutlierPointGenerator(PointGenerator):
    """Generates points with an outlier based on various base distributions"""
    
    def __init__(self, base_generator: PointGenerator, std_multiplier: float, noise_multiplier: Optional[float] = None):
        """
        Args:
            base_generator: The base point generator (correlation, function, or cluster)
            std_multiplier: How many standard deviations away the outlier should be.
            noise_multiplier: Original multiplier parameter (stored for metadata)
        """
        self.base_generator = base_generator
        self.std_multiplier = std_multiplier
        self.noise_multiplier = noise_multiplier  # Store original parameter
        self.outlier_point = None
        self.actual_distance = None
        self.distance_from_mean = None
    
    def generate_points(self, config: StimulusConfig) -> np.ndarray:
        axis_range = config.axis_range

        # Special handling for cluster-based outliers
        if isinstance(self.base_generator, SingleClusterPointGenerator):
            # For cluster outliers, generate base cluster points first
            cluster_points = self.base_generator.generate_points(config)
            distribution_points = cluster_points[:-1]  # All but the last point
            
            if len(distribution_points) == 0:
                self.outlier_point = np.random.uniform(axis_range[0] + 1.2, axis_range[1] - 0.2, 2)
                cluster_points[-1] = self.outlier_point
                self.actual_distance = 0
                self.distance_from_mean = 0
                return cluster_points
            
            # Get cluster center and std from the base generator
            cluster_center = self.base_generator.center
            cluster_std = self.base_generator.cluster_std
            
            # Calculate target distance: std_multiplier * cluster_std
            target_distance = self.std_multiplier * cluster_std
            
            # Generate outlier point at target distance from cluster center
            outlier = None
            attempts = 0
            max_attempts = 200
            current_distance = target_distance
            
            while outlier is None and attempts < max_attempts:
                # Random angle for direction
                angle = np.random.uniform(0, 2 * np.pi)
                
                # Place outlier at current_distance from cluster center
                candidate = cluster_center + current_distance * np.array([np.cos(angle), np.sin(angle)])
                
                if (axis_range[0] + 1.2 <= candidate[0] <= axis_range[1] - 0.2 and 
                    axis_range[0] + 1.2 <= candidate[1] <= axis_range[1] - 0.2):
                    outlier = candidate
                
                attempts += 1
                if attempts % 20 == 0:  # After 20 attempts at current distance
                    current_distance *= 0.95  # Shrink distance slightly
            
            # Fallback: try systematic search at reduced distances
            if outlier is None:
                for r in np.linspace(current_distance, cluster_std, 50):  # Don't go below cluster_std
                    found = False
                    for angle in np.linspace(0, 2 * np.pi, 16):
                        candidate = cluster_center + r * np.array([np.cos(angle), np.sin(angle)])
                        if (axis_range[0] + 1.2 <= candidate[0] <= axis_range[1] - 0.2 and 
                            axis_range[0] + 1.2 <= candidate[1] <= axis_range[1] - 0.2):
                            outlier = candidate
                            found = True
                            break
                    if found:
                        break
            
            # Final fallback if we still can't find a valid position
            if outlier is None:
                outlier = np.random.uniform(axis_range[0] + 1.2, axis_range[1] - 0.2, 2)
            
            # Store metadata
            self.outlier_point = outlier
            mean = np.mean(distribution_points, axis=0)
            self.distance_from_mean = np.linalg.norm(outlier - mean)
            self.actual_distance = np.min(np.linalg.norm(distribution_points - outlier, axis=1))
            
            # Replace last point with outlier
            cluster_points[-1] = outlier
            return cluster_points
        
        else:
            # Original approach for non-cluster outliers
            # Generate base points
            base_points = self.base_generator.generate_points(config)
            
            # The last point will be the outlier, the rest form the distribution
            distribution_points = base_points[:-1]

            if len(distribution_points) == 0:
                self.outlier_point = np.random.uniform(axis_range[0] + 1.2, axis_range[1] - 0.2, 2)
                base_points[-1] = self.outlier_point
                self.actual_distance = 0
                self.distance_from_mean = 0
                return base_points

            # Calculate mean and std of the base distribution
            mean = np.mean(distribution_points, axis=0)
            # Using mean of stds along each axis as a single metric
            std_dev = np.mean(np.std(distribution_points, axis=0))
            
            # Calculate target distance for the outlier
            target_distance = self.std_multiplier * std_dev
            
            # Find a valid position for the outlier
            outlier = None
            attempts = 0
            max_attempts = 200 # Safety break
            current_distance = target_distance
            
            while outlier is None and attempts < max_attempts:
                angle = np.random.uniform(0, 2 * np.pi)
                x = mean[0] + current_distance * np.cos(angle)
                y = mean[1] + current_distance * np.sin(angle)
                
                if axis_range[0] + 1.2 <= x <= axis_range[1] - 0.2 and axis_range[0] + 1.2 <= y <= axis_range[1] - 0.2:
                    outlier = np.array([x, y])
                
                attempts += 1
                if attempts % 20 == 0: # After 20 attempts at a radius
                    current_distance *= 0.95 # Shrink radius slightly
            
            if outlier is None: # Fallback if random placement fails
                for r in np.linspace(current_distance, 0, 100):
                    found = False
                    for angle in np.linspace(0, 2 * np.pi, 16):
                        x = mean[0] + r * np.cos(angle)
                        y = mean[1] + r * np.sin(angle)
                        if axis_range[0] + 1.2 <= x <= axis_range[1] - 0.2 and axis_range[0] + 1.2 <= y <= axis_range[1] - 0.2:
                            outlier = np.array([x, y])
                            found = True
                            break
                    if found:
                        break
            
            if outlier is None: # Final fallback
                outlier = np.random.uniform(axis_range[0] + 1.2, axis_range[1] - 0.2, 2)
            
            # Store metadata
            self.outlier_point = outlier
            self.distance_from_mean = np.linalg.norm(outlier - mean)
            if len(distribution_points) > 0:
                self.actual_distance = np.min(np.linalg.norm(distribution_points - outlier, axis=1))
            else:
                self.actual_distance = 0

            # Replace last point with outlier
            base_points[-1] = outlier
            
            return base_points
    
    def get_metadata(self) -> Dict:
        """Return metadata about the outlier parameters"""
        if self.outlier_point is None:
            raise ValueError("Points have not been generated yet")
            
        base_metadata = self.base_generator.get_metadata()
        metadata = {
            'stimulus_type': 'outlier',
            'base_distribution': base_metadata,
            'std_multiplier': self.std_multiplier,
            'actual_distance': self.actual_distance,  # The actual minimum distance achieved
            'distance_from_mean': self.distance_from_mean,  # The distance from distribution mean
            'outlier_point': self.outlier_point.tolist()
        }
        
        # Add noise_multiplier if it was provided
        if self.noise_multiplier is not None:
            metadata['noise_multiplier'] = self.noise_multiplier
            
        return metadata

    def sample_points(self, config: StimulusConfig, n_samples: int = 1) -> List[np.ndarray]:
        """Generate multiple samples of points with outliers"""
        return [self.generate_points(config) for _ in range(n_samples)]

    def to_string(self) -> str:
        """Return a string representation of the generator"""
        return f"outlier_{self.base_generator.to_string()}_std{self.std_multiplier}"

class StimulusPlotter:
    """Handles plotting of stimuli with segmentation"""
    
    def __init__(self):
        self.default_style = {
            'fontsize': 14,
            'linewidth': 0.5,
            'tick_length': 0.1,
            'dot_axis_offset': 0.225,
            'label_offset': 0.15,
            'axis_label_offset': 0.4,
            'output_size': (560, 560)
        }

    def _format_figure(self, fig: plt.Figure, x_offset: float = 0, y_offset: float = 0) -> np.ndarray:
        """Format a figure into a properly positioned image"""
        # Save figure to temporary file
        temp_path = f"temp_{uuid.uuid4()}.png"
        fig.savefig(temp_path)
        plt.close(fig)
        
        # Open and process image
        img = Image.open(temp_path).convert("RGB")
        
        # Get dimensions
        width, height = img.size
        max_dim = max(width, height)
        
        # Create square white background
        white_bg = Image.new('RGB', (max_dim, max_dim), 'white')
        
        # Calculate base centered position
        x_pos = (max_dim - width) // 2
        y_pos = (max_dim - height) // 2
        
        # Apply offsets
        x_pos += int((x_offset * (max_dim - width)) / 2)
        y_pos += int((y_offset * -1 * (max_dim - height)) / 2)
        
        # Ensure within bounds
        x_pos = max(0, min(x_pos, max_dim - width))
        y_pos = max(0, min(y_pos, max_dim - height))
        
        # Paste and resize
        white_bg.paste(img, (x_pos, y_pos))
        white_bg = white_bg.resize(self.default_style['output_size'], 
                                 resample=Image.Resampling.LANCZOS)
        
        # Clean up
        os.remove(temp_path)
        
        return white_bg

    def _convert_to_binary_mask(self, img: Image.Image) -> np.ndarray:
        """Convert image to binary mask where non-white pixels are 1"""
        # Convert to numpy array
        img_array = np.array(img)
        # White pixels are [255, 255, 255]
        mask = ~(img_array == 255).all(axis=2)
        return mask.astype(np.uint8)

    def _setup_figure(self, axis_config: str) -> Tuple[plt.Figure, plt.Axes]:
        """Create figure with appropriate size based on axis configuration"""
        sizes = {
            "xy": (8, 8),
            "x": (8, 2),
            "y": (2, 8)
        }
        fig, ax = plt.subplots(figsize=sizes[axis_config])
        return fig, ax

    def _create_segmentation_figures(self, axis_config: str, n_points: int, axis_range: Tuple[int, int]) -> Dict:
        """Create all necessary figures for segmentation"""
        range_size = axis_range[1] - axis_range[0] + 1
        
        figs = {
            'main': self._setup_figure(axis_config),
            'dots': [self._setup_figure(axis_config) for _ in range(n_points)]
        }
        
        if axis_config == "xy":
            for axis in ['x', 'y']:
                figs.update({
                    f'ticks_{axis}': [self._setup_figure(axis_config) for _ in range(range_size)],
                    f'tick_labels_{axis}': [self._setup_figure(axis_config) for _ in range(range_size)],
                    f'axis_{axis}': self._setup_figure(axis_config),
                    f'axis_label_{axis}': self._setup_figure(axis_config)
                })
        else:
            figs.update({
                'ticks': [self._setup_figure(axis_config) for _ in range(range_size)],
                'tick_labels': [self._setup_figure(axis_config) for _ in range(range_size)],
                'axis': self._setup_figure(axis_config),
                'axis_label': self._setup_figure(axis_config)
            })
        
        return figs

    def _setup_axis_style(self, ax: plt.Axes, axis_range: Tuple[int, int], axis_config: str):
        """Apply common axis styling"""
        # Add padding to prevent dots from being cut off
        padding = 0.5  # Increase this value if needed for larger dots
        # Set limits
        if axis_config == "xy":
            ax.set_xlim(axis_range[0] - padding, axis_range[1] + padding)
            ax.set_ylim(axis_range[0] - padding, axis_range[1] + padding)
        elif axis_config == "x":
            ax.set_xlim(axis_range[0] - padding, axis_range[1] + padding)
            ax.set_ylim(-0.5, 0.5)
        else:  # y
            ax.set_xlim(-0.5, 0.5)
            ax.set_ylim(axis_range[0] - padding, axis_range[1] + padding)

        # Remove spines and ticks
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor('white')

    def _plot_axis_lines(self, ax: plt.Axes, axis_config: str, color: str = 'k'):
        """Plot axis lines based on configuration"""
        if axis_config in ["xy", "x"]:
            ax.axhline(y=0, color=color, linestyle='-', 
                      linewidth=self.default_style['linewidth'], zorder=1)
        if axis_config in ["xy", "y"]:
            ax.axvline(x=0, color=color, linestyle='-', 
                      linewidth=self.default_style['linewidth'], zorder=1)

    def _plot_ticks_and_labels(self, ax: plt.Axes, x_ticks: Optional[np.ndarray],
                              x_tick_labels: Optional[List[str]], y_ticks: Optional[np.ndarray],
                              y_tick_labels: Optional[List[str]], axis_config: str,
                              color: str = 'k'):
        """Plot tick marks and labels"""
        tick_length = self.default_style['tick_length']

        if axis_config in ["xy", "x"] and x_ticks is not None and x_tick_labels is not None:
            ax.vlines(x_ticks, -tick_length, tick_length, colors=color,
                     linewidth=self.default_style['linewidth'], zorder=1)
            for tick, label in zip(x_ticks, x_tick_labels):
                ax.text(tick, -self.default_style['label_offset'], str(label),
                       ha='center', va='top', fontsize=self.default_style['fontsize'])

        if axis_config in ["xy", "y"] and y_ticks is not None and y_tick_labels is not None:
            ax.hlines(y_ticks, -tick_length, tick_length, colors=color,
                     linewidth=self.default_style['linewidth'], zorder=1)
            for tick, label in zip(y_ticks, y_tick_labels):
                ax.text(-self.default_style['label_offset'], tick, str(label),
                       ha='right', va='center', fontsize=self.default_style['fontsize'])

    def plot_stimulus(self, points: np.ndarray, config: StimulusConfig, 
                     x_offset: float = 0, y_offset: float = 0, n_ticks=None, tick_scale=None) -> Tuple[Image.Image, Dict]:
        """Plot main stimulus and generate segmentation masks"""
        # Create and plot main figure
        main_fig, main_ax = self._setup_figure(config.axis_config)
        self._setup_axis_style(main_ax, config.axis_range, config.axis_config)
        
        chart_type = getattr(config, 'chart_type', 'scatter')
        orientation = getattr(config, 'spatial_config', 'x')

        # Plot main content
        if config.axis_config == "xy":
            x, y = points[:, 0], points[:, 1]
        else:
            x = points if config.axis_config == "x" else np.zeros_like(points) + self.default_style['dot_axis_offset']
            y = points if config.axis_config == "y" else np.zeros_like(points) + self.default_style['dot_axis_offset']

        while len(x) > len(config.plot_dot_size):
            config.plot_dot_size.append(config.plot_dot_size[-1])
        while len(x) > len(config.plot_dot_shape):
            config.plot_dot_shape.append(config.plot_dot_shape[-1])

        if chart_type == 'bar':
            if orientation == 'y':
                y = np.arange(config.n_points)
                x = points[:, 0]
                main_ax.barh(y, x, color=config.plot_color, height=0.6, zorder=2)
            else:
                x = np.arange(config.n_points)
                y = points[:, 1]
                main_ax.bar(x, y, color=config.plot_color, width=0.6, zorder=2)
        elif chart_type == 'line':
            sort_idx = np.argsort(x)
            x_sorted, y_sorted = x[sort_idx], y[sort_idx]
            main_ax.plot(x_sorted, y_sorted, color=config.plot_color[0], linewidth=2, zorder=2)
            for i, (xi, yi) in enumerate(zip(x, y)):
                main_ax.scatter(xi, yi, c=config.plot_color[i], s=config.plot_dot_size[i],
                              marker=config.plot_dot_shape[i], zorder=3)
        else:
            for i, (xi, yi) in enumerate(zip(x, y)):
                main_ax.scatter(xi, yi, c=config.plot_color[i], s=config.plot_dot_size[i],
                              marker=config.plot_dot_shape[i], zorder=2)

        self._plot_axis_lines(main_ax, config.axis_config)
        if n_ticks is None:
            n_ticks = min(config.axis_range[1], 8)
        if tick_scale is None:
            tick_scale = config.axis_range[1]
        # Don't increment tick_scale when it's passed as parameter
        if tick_scale == 9:
            tick_scale = 8
        assert tick_scale % n_ticks == 0
        assert config.axis_range[1] % n_ticks == 0
        tick_step = tick_scale // n_ticks

        numeric_ticks = np.arange(config.axis_range[0], config.axis_range[1] + 1, step=config.axis_range[1] // n_ticks)
        numeric_labels = np.arange(config.axis_range[0], tick_scale + 1, step=tick_step)

        x_ticks, x_tick_labels = None, None
        y_ticks, y_tick_labels = None, None

        if chart_type == 'bar' and orientation == 'y':
            x_ticks = numeric_ticks
            x_tick_labels = config.x_tick_labels or numeric_labels
            y_ticks = np.arange(config.n_points)
            y_tick_labels = config.y_tick_labels or y_ticks
            padding = 0.5
            main_ax.set_xlim(config.axis_range[0] - padding, config.axis_range[1] + padding)
            main_ax.set_ylim(-0.5, config.n_points - 0.5)
        elif chart_type == 'bar':
            x_ticks = np.arange(config.n_points)
            x_tick_labels = config.x_tick_labels or x_ticks
            y_ticks = numeric_ticks
            y_tick_labels = config.y_tick_labels or numeric_labels
            padding = 0.5
            main_ax.set_xlim(-0.5, config.n_points - 0.5)
            main_ax.set_ylim(config.axis_range[0] - padding, config.axis_range[1] + padding)
        else:
            ticks = numeric_ticks
            tick_labels = config.tick_labels or numeric_labels
            if config.axis_config in ['xy', 'x']:
                x_ticks = ticks
                x_tick_labels = tick_labels
            if config.axis_config in ['xy', 'y']:
                y_ticks = ticks
                y_tick_labels = tick_labels

        def _apply_bar_axis_limits(ax: plt.Axes):
            padding = 0.5
            if chart_type == 'bar':
                if orientation == 'y':
                    ax.set_xlim(config.axis_range[0] - padding, config.axis_range[1] + padding)
                    ax.set_ylim(-0.5, config.n_points - 0.5)
                else:
                    ax.set_xlim(-0.5, config.n_points - 0.5)
                    ax.set_ylim(config.axis_range[0] - padding, config.axis_range[1] + padding)

        self._plot_ticks_and_labels(main_ax, x_ticks, x_tick_labels, y_ticks, y_tick_labels, config.axis_config)
        _apply_bar_axis_limits(main_ax)
        
        # Format main figure
        main_image = self._format_figure(main_fig, x_offset, y_offset)
        
        # Process segmentation figures one at a time
        seg_dict = {}

        # Compute background mask
        seg_dict['bg'] = [~self._convert_to_binary_mask(main_image) // 255]
        
        # Process dots
        seg_dict['dots'] = []
        chart_type = getattr(config, 'chart_type', 'scatter')
        for i, (xi, yi) in enumerate(zip(x, y)):
            fig, ax = self._setup_figure(config.axis_config)
            self._setup_axis_style(ax, config.axis_range, config.axis_config)
            _apply_bar_axis_limits(ax)
            if chart_type == 'bar':
                if orientation == 'y':
                    ax.barh(yi, xi, color='r', height=0.6, zorder=2)
                else:
                    ax.bar(xi, yi, color='r', width=0.6, zorder=2)
            else:
                ax.scatter(xi, yi, c='r', s=config.plot_dot_size[i], marker=config.plot_dot_shape[i], zorder=2)
            img = self._format_figure(fig, x_offset, y_offset)
            seg_dict['dots'].append(self._convert_to_binary_mask(img))
        
        # Process ticks, labels, and axes based on configuration
        if config.axis_config == "xy":
            for axis in ['x', 'y']:
                seg_dict[f'ticks_{axis}'] = []
                seg_dict[f'tick_labels_{axis}'] = []

                # Process ticks and labels
                axis_ticks = x_ticks if axis == 'x' else y_ticks
                axis_labels = x_tick_labels if axis == 'x' else y_tick_labels
                if axis_ticks is None or axis_labels is None:
                    continue
                for tick, label in zip(axis_ticks, axis_labels):
                    # Tick marks
                    fig, ax = self._setup_figure(config.axis_config)
                    self._setup_axis_style(ax, config.axis_range, config.axis_config)
                    _apply_bar_axis_limits(ax)
                    if axis == 'x':
                        ax.vlines(tick, -self.default_style['tick_length'],
                                self.default_style['tick_length'], colors='r',
                                linewidth=self.default_style['linewidth'], zorder=1)
                    else:
                        ax.hlines(tick, -self.default_style['tick_length'], 
                                self.default_style['tick_length'], colors='r', 
                                linewidth=self.default_style['linewidth'], zorder=1)
                    img = self._format_figure(fig, x_offset, y_offset)
                    seg_dict[f'ticks_{axis}'].append(self._convert_to_binary_mask(img))
                    
                    # Tick labels
                    fig, ax = self._setup_figure(config.axis_config)
                    self._setup_axis_style(ax, config.axis_range, config.axis_config)
                    _apply_bar_axis_limits(ax)
                    if axis == 'x':
                        ax.text(tick, -self.default_style['label_offset'], str(label),
                               ha='center', va='top', fontsize=self.default_style['fontsize'],
                               color='r')
                    else:
                        ax.text(-self.default_style['label_offset'], tick, str(label),
                               ha='right', va='center', fontsize=self.default_style['fontsize'],
                               color='r')
                    img = self._format_figure(fig, x_offset, y_offset)
                    seg_dict[f'tick_labels_{axis}'].append(self._convert_to_binary_mask(img))
                
                # Process axis lines and labels
                seg_dict[f'axis_{axis}'] = []
                fig, ax = self._setup_figure(config.axis_config)
                self._setup_axis_style(ax, config.axis_range, config.axis_config)
                _apply_bar_axis_limits(ax)
                self._plot_axis_lines(ax, axis, color='r')
                img = self._format_figure(fig, x_offset, y_offset)
                seg_dict[f'axis_{axis}'].append(self._convert_to_binary_mask(img))
        else:
            # Similar process for single axis configuration
            seg_dict['ticks'] = []
            seg_dict['tick_labels'] = []

            ticks = x_ticks if config.axis_config == 'x' else y_ticks
            labels = x_tick_labels if config.axis_config == 'x' else y_tick_labels
            if ticks is None:
                ticks = []
                labels = []

            for tick, label in zip(ticks, labels):  # Process ticks and labels (similar to above but for single axis)
                # Tick marks
                fig, ax = self._setup_figure(config.axis_config)
                self._setup_axis_style(ax, config.axis_range, config.axis_config)
                _apply_bar_axis_limits(ax)
                if config.axis_config == 'x':
                    ax.vlines(tick, -self.default_style['tick_length'],
                            self.default_style['tick_length'], colors='r',
                            linewidth=self.default_style['linewidth'], zorder=1)
                else:  # y axis
                    ax.hlines(tick, -self.default_style['tick_length'], 
                            self.default_style['tick_length'], colors='r', 
                            linewidth=self.default_style['linewidth'], zorder=1)
                img = self._format_figure(fig, x_offset, y_offset)
                seg_dict['ticks'].append(self._convert_to_binary_mask(img))
                
                # Tick labels
                fig, ax = self._setup_figure(config.axis_config)
                self._setup_axis_style(ax, config.axis_range, config.axis_config)
                _apply_bar_axis_limits(ax)
                if config.axis_config == 'x':
                    ax.text(tick, -self.default_style['label_offset'], str(label),
                           ha='center', va='top', fontsize=self.default_style['fontsize'],
                           color='r')
                else:  # y axis
                    ax.text(-self.default_style['label_offset'], tick, str(label),
                           ha='right', va='center', fontsize=self.default_style['fontsize'],
                           color='r')
                img = self._format_figure(fig, x_offset, y_offset)
                seg_dict['tick_labels'].append(self._convert_to_binary_mask(img))
            
            # Process axis line and label
            seg_dict['axis'] = []
            fig, ax = self._setup_figure(config.axis_config)
            self._setup_axis_style(ax, config.axis_range, config.axis_config)
            self._plot_axis_lines(ax, config.axis_config, color='r')
            img = self._format_figure(fig, x_offset, y_offset)
            seg_dict['axis'].append(self._convert_to_binary_mask(img))

        return main_image, seg_dict
    
    def plot_stimulus_without_segmentation(
        self, 
        points: np.ndarray, 
        config: StimulusConfig,
        x_offset: float = 0, 
        y_offset: float = 0
    ) -> Image.Image:
        """Plot main stimulus without generating segmentation masks"""
        # Create and plot main figure
        main_fig, main_ax = self._setup_figure(config.axis_config)
        self._setup_axis_style(main_ax, config.axis_range, config.axis_config)
        
        # Plot main content
        if config.axis_config == "xy":
            x, y = points[:, 0], points[:, 1]
        else:
            x = points if config.axis_config == "x" else np.zeros_like(points) + self.default_style['dot_axis_offset']
            y = points if config.axis_config == "y" else np.zeros_like(points) + self.default_style['dot_axis_offset']

        # Plot dots or chart elements
        if len(config.plot_dot_size) != len(x):
            config.plot_dot_size = [config.plot_dot_size[0]] * len(x)
        chart_type = getattr(config, 'chart_type', 'scatter')
        if chart_type == 'bar':
            main_ax.bar(x, y, color=config.plot_color, width=0.6, zorder=2)
        elif chart_type == 'line':
            sort_idx = np.argsort(x)
            x_sorted, y_sorted = x[sort_idx], y[sort_idx]
            main_ax.plot(x_sorted, y_sorted, color=config.plot_color[0], linewidth=2, zorder=2)
            for i, (xi, yi) in enumerate(zip(x, y)):
                main_ax.scatter(xi, yi, c=config.plot_color[i], s=config.plot_dot_size[i],
                            marker=config.plot_dot_shape[i], zorder=3)
        else:
            for i, (xi, yi) in enumerate(zip(x, y)):
                main_ax.scatter(xi, yi, c=config.plot_color[i], s=config.plot_dot_size[i],
                            marker=config.plot_dot_shape[i], zorder=2)

        # Plot axes and ticks
        self._plot_axis_lines(main_ax, config.axis_config)
        ticks = np.arange(config.axis_range[0], config.axis_range[1] + 1, step=config.axis_range[1] // config.n_ticks)
        tick_step = config.tick_scale // config.n_ticks
        tick_labels = np.arange(config.axis_range[0], config.tick_scale + 1, step=tick_step)
        self._plot_ticks_and_labels(main_ax, ticks, config.tick_labels or tick_labels, config.axis_config)
        
        # Format and return main figure
        return self._format_figure(main_fig, x_offset, y_offset)

    def _create_segmentation_dict(self, figs: Dict, axis_config: str) -> Dict:
        """Create dictionary of segmentation figures"""
        if axis_config == "xy":
            return {
                "dots": [fig for fig, _ in figs['dots']],
                "ticks": {
                    "x": [fig for fig, _ in figs['ticks_x']],
                    "y": [fig for fig, _ in figs['ticks_y']]
                },
                "tick_labels": {
                    "x": [fig for fig, _ in figs['tick_labels_x']],
                    "y": [fig for fig, _ in figs['tick_labels_y']]
                },
                "axis": {
                    "x": [figs['axis_x'][0]],
                    "y": [figs['axis_y'][0]]
                },
                "axis_label": {
                    "x": [figs['axis_label_x'][0]],
                    "y": [figs['axis_label_y'][0]]
                }
            }
        else:
            return {
                "dots": [fig for fig, _ in figs['dots']],
                "ticks": {axis_config: [fig for fig, _ in figs['ticks']]},
                "tick_labels": {axis_config: [fig for fig, _ in figs['tick_labels']]},
                "axis": {axis_config: [figs['axis'][0]]},
                "axis_label": {axis_config: [figs['axis_label'][0]]}
            }


class StimulusGenerator:
    """Main class for generating and saving stimuli"""
    
    def __init__(
        self,
        point_generator: PointGenerator,
        output_dir: str = "stimuli",
        metadata_dir: str = "metadata",
        segmentation_dir: str = "segmentation",
        root_dir: str = "datasets/new_dataset",
        generate_segmentation: bool = True,
        save_files: bool = True
    ):
        self.point_generator = point_generator
        self.output_dir = root_dir + "/" + output_dir
        self.metadata_dir = root_dir + "/" + metadata_dir
        self.segmentation_dir = root_dir + "/" + segmentation_dir
        self.plotter = StimulusPlotter()
        self.root_dir = root_dir
        self.generate_segmentation = generate_segmentation
        self.save_files = save_files
        self.base_colors = ["red", "blue", "green", "orange"]
        self.additional_colors = ["purple", "brown", "cyan", "magenta"]

        # Create directories if they don't exist
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.metadata_dir, exist_ok=True)
        if self.generate_segmentation:
            os.makedirs(self.segmentation_dir, exist_ok=True)

    def _compute_stats(self, points: np.ndarray, config: StimulusConfig) -> Dict:
        """Compute statistics about the stimulus"""
        if config.axis_config == "xy":
            return {
                "mean": np.mean(points, axis=0),
                "median": np.median(points, axis=0),
                "std": np.std(points, axis=0),
                "min": np.min(points, axis=0),
                "max": np.max(points, axis=0)
            }
        else:
            return {
                "mean": float(np.mean(points)),
                "median": float(np.median(points)),
                "std": float(np.std(points)),
                "min": float(np.min(points)),
                "max": float(np.max(points))
            }
        
    def _compute_distances(self, points: np.ndarray, config: StimulusConfig) -> Dict:
        """Compute distances between points"""
        if config.axis_config == "xy":
            # Create n_points x n_points x 2 matrix for x and y distances
            n = len(points)
            distances = np.zeros((n, n, 2))
            
            # Calculate x and y distances
            x_dists = np.abs(points[:, 0, None] - points[None, :, 0])
            y_dists = np.abs(points[:, 1, None] - points[None, :, 1])
            
            # Fill in the matrix
            distances[:, :, 0] = x_dists  # x distances
            distances[:, :, 1] = y_dists  # y distances
        else:
            # For single axis, just calculate absolute differences
            n = len(points)
            distances = np.zeros((n, n))
            for i in range(n):
                for j in range(n):  # Full matrix
                    distances[i, j] = abs(points[i] - points[j])

        return distances

    def _generate_chart_colors(self, config: StimulusConfig) -> None:
        """Apply chart-type-specific color rules."""
        full_palette = self.base_colors + self.additional_colors

        if config.chart_type == 'bar':
            if config.n_points > len(full_palette):
                raise ValueError("Bar charts require a unique color per bar but not enough unique colors are available")
            sampled_colors = list(np.random.choice(full_palette, size=config.n_points, replace=False))
            config.plot_color = sampled_colors
            config.color = sampled_colors
        elif config.chart_type == 'line':
            line_color = np.random.choice(full_palette)
            config.plot_color = [line_color for _ in range(config.n_points)]
            config.color = config.plot_color
        else:
            if isinstance(config.color, str):
                config.plot_color = [config.color] * config.n_points
            else:
                config.plot_color = config.color
        config.color = config.plot_color

    def _assign_bar_tick_labels(self, config: StimulusConfig) -> None:
        """Assign categorical tick labels for bar charts."""
        if config.chart_type != 'bar':
            return

        available_labels = list(string.ascii_uppercase)
        if config.n_points > len(available_labels):
            raise ValueError("Not enough unique labels available for bar chart categories")

        sampled_labels = list(np.random.choice(available_labels, size=config.n_points, replace=False))
        if getattr(config, 'spatial_config', 'x') == 'y':
            config.y_tick_labels = sampled_labels
            config.x_tick_labels = None
        else:
            config.x_tick_labels = sampled_labels
            config.y_tick_labels = None

    def _assign_line_shapes(self, config: StimulusConfig) -> None:
        """Ensure line charts have distinguishable markers despite a single color."""
        if config.chart_type != 'line':
            return

        shape_pool = ['o', '^', '*', 's']
        shapes = config.dot_shape if isinstance(config.dot_shape, list) else [config.dot_shape] * config.n_points

        while len(shapes) < config.n_points:
            shapes.append(shapes[-1])

        if len(set(shapes)) == 1:
            multiplier = (config.n_points + len(shape_pool) - 1) // len(shape_pool)
            shapes = (shape_pool * multiplier)[:config.n_points]
            np.random.shuffle(shapes)

        config.dot_shape = shapes
        config.plot_dot_shape = shapes

    def _validate_chart_constraints(self, points: np.ndarray, config: StimulusConfig) -> np.ndarray:
        """Enforce chart specific constraints such as unique x-values."""
        if config.chart_type not in ['bar', 'line']:
            return points

        if not (2 <= config.n_points <= 8):
            raise ValueError("Bar and line charts must have between 2 and 8 elements")

        # Bar charts use categorical axes and do not require unique numeric positions
        if config.chart_type == 'bar':
            return points

        max_attempts = 50
        for _ in range(max_attempts):
            x_values = points[:, 0] if config.axis_config == 'xy' else points
            if len(np.unique(x_values)) == config.n_points:
                return points
            points = self.point_generator.generate_points(config)

        # As a final fallback, force unique x-values using visible tick positions
        grid_interval = max(1, config.axis_range[1] // config.n_ticks)
        candidate_positions = list(range(config.axis_range[0], config.axis_range[1] + 1, grid_interval))

        # Avoid plotting on the axis if the lower bound is 0 and we have enough room
        if config.axis_range[0] == 0 and len(candidate_positions) > config.n_points:
            candidate_positions = [pos for pos in candidate_positions if pos != 0]

        if len(candidate_positions) < config.n_points:
            raise ValueError("Unable to generate enough unique x-values for the requested chart")

        np.random.shuffle(candidate_positions)
        unique_x = candidate_positions[: config.n_points]

        # Replace the x-values while keeping y-values untouched
        if config.axis_config == 'xy':
            points[:, 0] = unique_x
        else:
            points = np.array(unique_x).reshape(-1, 1)

        return points

    def _prepare_bar_points(self, points: np.ndarray, config: StimulusConfig) -> np.ndarray:
        """Map bar chart points onto categorical and numeric axes."""
        if config.chart_type != 'bar':
            return points

        orientation = getattr(config, 'spatial_config', 'x')
        numeric_values = points[:, 1] if points.ndim > 1 else points
        secondary_values = points[:, 0] if points.ndim > 1 else points

        if orientation == 'y':
            categories = np.arange(config.n_points)
            numeric = secondary_values[: config.n_points]
            return np.column_stack((numeric, categories))

        categories = np.arange(config.n_points)
        numeric = numeric_values[: config.n_points]
        return np.column_stack((categories, numeric))
    
    def _is_collinear(self, points: np.ndarray) -> Dict[str, bool]:
        """
        Check if points are collinear along x or y axis.
        Returns dict with 'x_collinear' and 'y_collinear' bools.
        """
        if isinstance(points, list):
            points = np.array(points)
        if len(points.shape) == 1:
            # Single point is technically collinear
            return '0d'

        if len(np.unique(points[:, 0])) == 1:
            return 'x_collinear'
        elif len(np.unique(points[:, 1])) == 1:
            return 'y_collinear'
        else:
            return '2d'
    
    def generate_stimulus(self, config: StimulusConfig) -> Dict:
        """Generate a single stimulus and its metadata
        
        Args:
            config: Configuration for the stimulus
        
        Returns:
            Dict containing metadata about the generated stimulus
        """
        
        # Generate unique ID
        stimulus_id = str(uuid.uuid4())

        # Apply chart-specific color logic
        self._generate_chart_colors(config)
        self._assign_line_shapes(config)
        self._assign_bar_tick_labels(config)

        # Generate points *on the absolute grid*
        points = self.point_generator.generate_points(config)
        points = self._prepare_bar_points(points, config)
        points = self._validate_chart_constraints(points, config)

        # ------------------------------------------------------------------
        # Convert absolute grid points to *relative* positions according to
        # the tick labelling scheme.  This mapping is linear because ticks
        # are equally spaced in the underlying grid.
        # ------------------------------------------------------------------
        tick_step = max(1, config.tick_scale // config.n_ticks)

        # Helper to transform coordinates
        def _to_relative(coord):
            return (coord * config.tick_scale) / config.axis_range[1]
        
        def _rounded_relative(coord):
            round_neighborhood = max(1, config.tick_scale // (config.n_ticks*2))
            
            # Round coord up or down within round_neighborhood
            if isinstance(coord, (int, float)):
                # For single value
                rounded = round(coord / round_neighborhood) * round_neighborhood
                return rounded
            else:
                # For array-like coordinates
                rounded = np.round(coord / round_neighborhood) * round_neighborhood
                return rounded

        if config.axis_config == "xy":
            relative_points = np.column_stack((_to_relative(points[:, 0]), _to_relative(points[:, 1])))
            rounded_relative_points = np.column_stack((_rounded_relative(relative_points[:, 0]), _rounded_relative(relative_points[:, 1])))
        else:
            relative_points = _to_relative(points)
            rounded_relative_points = _rounded_relative(relative_points)

        # Plot stimulus and optionally generate segmentation masks
        if self.generate_segmentation:
            main_image, seg_dict = self.plotter.plot_stimulus(
                points=points,
                config=config,
                x_offset=config.x_offset,
                y_offset=config.y_offset,
                n_ticks=config.n_ticks,
                tick_scale=config.tick_scale
            )
            # Save segmentation masks
            if self.save_files:
                segmentation_path = f"{self.segmentation_dir}/{stimulus_id}.npy"
                np.save(segmentation_path, seg_dict, allow_pickle=True)
            else:
                segmentation_path = None
        else:
            main_image = self.plotter.plot_stimulus_without_segmentation(
                points=points,
                config=config,
                x_offset=config.x_offset,
                y_offset=config.y_offset
            )
            seg_dict = None
            segmentation_path = None

        # Save main stimulus image
        if self.save_files:
            image_path = f"{self.output_dir}/{stimulus_id}.png"
            main_image.save(image_path)
        else:
            image_path = None
        
        # Create and save metadata
        stats = self._compute_stats(points, config)
        dimensionality = self._is_collinear(points)
        distances = self._compute_distances(points, config)
        metadata_path = f"{self.metadata_dir}/{stimulus_id}.npy"
        
        # Get point generator metadata
        generator_metadata = self.point_generator.get_metadata()
        
        # Create visual identity metadata
        # TODO: The following code is only compatible with ensemble generation 
        visual_identity = {}
        if generator_metadata['stimulus_type'] == 'cluster':
            # For clusters, we have two cluster identities and one query identity
            n_per_cluster = generator_metadata['n_points_per_cluster']
            visual_identity = {
                'cluster_1': {
                    'color': config.plot_color[:n_per_cluster[0]],
                    'shape': config.plot_dot_shape[:n_per_cluster[0]]
                },
                'cluster_2': {
                    'color': config.plot_color[n_per_cluster[0]:sum(n_per_cluster)],
                    'shape': config.plot_dot_shape[n_per_cluster[0]:sum(n_per_cluster)]
                },
                'query_point': {
                    'color': config.plot_color[-1],
                    'shape': config.plot_dot_shape[-1]
                }
            }
        elif generator_metadata['stimulus_type'] == 'outlier' and generator_metadata['base_distribution']['stimulus_type'] == 'single_cluster':
            # For outlier cluster stimuli, we have one cluster and one outlier
            visual_identity = {
                'cluster': {
                    'color': config.plot_color[:-1],  # All but last point
                    'shape': config.plot_dot_shape[:-1]
                },
                'outlier': {
                    'color': config.plot_color[-1],  # Last point
                    'shape': config.plot_dot_shape[-1]
                }
            }
        else:
            # For other types, all points have same identity
            visual_identity = {
                'color': config.plot_color[0] if isinstance(config.plot_color, list) else config.plot_color,
                'shape': config.plot_dot_shape[0] if isinstance(config.plot_dot_shape, list) else config.plot_dot_shape
            }
        
        default_tick_labels = np.arange(config.axis_range[0], config.tick_scale + 1, step=tick_step)

        def _as_list(labels):
            if labels is None:
                return None
            return labels.tolist() if isinstance(labels, np.ndarray) else list(labels)

        x_tick_labels = config.x_tick_labels
        y_tick_labels = config.y_tick_labels

        if x_tick_labels is None and config.axis_config in ['xy', 'x']:
            x_tick_labels = config.tick_labels or default_tick_labels
        if y_tick_labels is None and config.axis_config in ['xy', 'y']:
            y_tick_labels = config.tick_labels or default_tick_labels

        tick_labels = {
            "x": _as_list(x_tick_labels) if x_tick_labels is not None else [],
            "y": _as_list(y_tick_labels) if y_tick_labels is not None else []
        }

        metadata = {
            "id": stimulus_id,
            "grid_points": points.tolist(),
            "relative_points": relative_points.tolist(),
            "rounded_relative_points": rounded_relative_points.tolist(),
            "point_generator": self.point_generator.to_string(),
            "generator_metadata": generator_metadata,
            "visual_identity": visual_identity,
            "image_path": image_path,
            "segmentation_path": segmentation_path,
            "metadata_path": metadata_path,
            "distances": distances,
            "dimensionality": dimensionality,
            **stats,
            **config.__dict__,
            "tick_labels": tick_labels
        }
        
        if self.save_files:
            np.save(metadata_path, metadata, allow_pickle=True)
            return metadata
        return main_image, seg_dict, metadata
    
    def generate_dataset(self, configs: List[StimulusConfig], n_samples: int = 1) -> List[Dict]:
        """Generate multiple stimuli from a list of configurations"""
        return [[self.generate_stimulus(config) for _ in range(n_samples)] for config in configs]


class ConfigurableDatasetGenerator:
    """Generates balanced datasets based on YAML configuration"""
    
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.parameter_spaces = self.config['parameter_spaces']
        
    def get_dot_objects(self) -> List[Tuple[str, str]]:
        """Generate all possible color-shape combinations"""
        colors = self.parameter_spaces['colors']
        shapes = self.parameter_spaces['shapes']
        return list(itertools.product(colors, shapes))
    
    def get_grid_positions(self, tick_config: Dict = None) -> List[Tuple[int, int]]:
        """Generate all grid positions based on tick configuration"""
        if tick_config is None:
            # Use default tick config if none provided
            tick_config = self.parameter_spaces['tick_configs'][0]
        
        axis_range = tick_config['axis_range']
        max_pos = axis_range[1]
        return [(x, y) for x in range(1, max_pos + 1) for y in range(1, max_pos + 1)]
    
    def get_disjoint_dot_pairs(self, dot_objects: List[Tuple[str, str]], n_pairs: int) -> List[List[Tuple[str, str]]]:
        """Generate disjoint pairs of dot objects"""
        all_pairs = list(itertools.combinations(dot_objects, 2))
        np.random.shuffle(all_pairs)
        
        disjoint_pairs = []
        used_dots = set()
        
        for pair in all_pairs:
            if len(disjoint_pairs) == n_pairs:
                break
            if not (pair[0] in used_dots or pair[1] in used_dots):
                disjoint_pairs.append(pair)
                used_dots.add(pair[0])
                used_dots.add(pair[1])
        
        return disjoint_pairs
    
    def generate_buddy_samples(self, dot_objects: List[Tuple[str, str]], main_dot: Tuple[str, str], 
                             n_buddies: int, n_samples: int) -> List[List[Tuple[str, str]]]:
        """Generate multiple samples of buddy dots for a main dot"""
        other_dots = [d for d in dot_objects if d != main_dot]
        buddy_samples = []
        for _ in range(n_samples):
            buddies = list(np.random.choice(range(len(other_dots)), size=n_buddies, replace=False))
            buddy_samples.append([main_dot] + [other_dots[i] for i in buddies])
        return buddy_samples
    
    def calculate_combinations(self, balance_config: List, additional_params: Dict = None) -> int:
        """Calculate total number of combinations from balance configuration"""
        total_combinations = 1
        
        # Get tick config for position calculations
        tick_config_idx = 0
        if additional_params and 'tick_configs' in additional_params:
            tick_config_idx = additional_params['tick_configs'][0]
        tick_config = self.parameter_spaces['tick_configs'][tick_config_idx]
        
        for item in balance_config:
            if isinstance(item, str):
                if item == "dot_objects":
                    total_combinations *= len(self.get_dot_objects())
                elif item == "grid_positions":
                    total_combinations *= len(self.get_grid_positions(tick_config))
                elif item == "position_pairs":
                    positions = self.get_grid_positions(tick_config)
                    total_combinations *= len(list(itertools.combinations(positions, 2)))
                elif item == "main_dot_objects":
                    total_combinations *= len(self.get_dot_objects())
                elif item.startswith("tick_configs"):
                    total_combinations *= len(self.parameter_spaces['tick_configs'])
            elif isinstance(item, dict):
                for key, value in item.items():
                    if key == "point_generators":
                        total_combinations *= len(value)
                    elif key == "disjoint_dot_pairs":
                        total_combinations *= value
                    elif key == "buddy_samples":
                        total_combinations *= value
                    elif key == "spatial_configs":
                        total_combinations *= len(value)
                    elif key == "n_points":
                        total_combinations *= len(value) if isinstance(value, list) else 1
        
        return total_combinations
    
    def expand_parameters(self, balance_config: List, current_combinations: int, target_count: int) -> List:
        """Expand parameters when combinations are insufficient"""
        expansion_rules = self.config.get('expansion_rules', {})
        priority_axes = expansion_rules.get('priority_axes', [])
        
        expanded_config = balance_config.copy()
        
        for axis in priority_axes:
            if current_combinations >= target_count:
                break
                
            if isinstance(axis, str):
                if axis == "tick_configs":
                    expanded_config.append({"tick_configs": list(range(len(self.parameter_spaces['tick_configs'])))})
                    current_combinations *= len(self.parameter_spaces['tick_configs'])
            elif isinstance(axis, dict):
                for param_name, param_values in axis.items():
                    expanded_config.append({param_name: param_values})
                    current_combinations *= len(param_values)
                    if current_combinations >= target_count:
                        break
        
        return expanded_config
    
    def subsample_combinations(self, all_combinations: List, target_count: int, strategy: str = "random") -> List:
        """Subsample combinations when there are too many"""
        if len(all_combinations) <= target_count:
            return all_combinations
            
        if strategy == "random":
            return list(np.random.choice(len(all_combinations), size=target_count, replace=False))
        elif strategy == "stratified":
            # Simple stratified sampling - divide into bins and sample from each
            step = len(all_combinations) // target_count
            indices = list(range(0, len(all_combinations), step))[:target_count]
            return indices
        else:
            return list(range(target_count))
    
    def generate_stimulus_configs_for_n(self, n_config: Dict, n_points: int) -> List[StimulusConfig]:
        """Generate stimulus configurations for a specific n value"""
        configs = []
        target_count = n_config['target_count']
        balance_across = n_config['balance_across']
        additional_params = n_config.get('additional_params', {})
        chart_type = n_config.get('chart_type', 'scatter')

        if chart_type in ['bar', 'line'] and not (2 <= n_points <= 8):
            raise ValueError("Bar and line charts must specify between 2 and 8 points")

        # Calculate total combinations
        total_combinations = self.calculate_combinations(balance_across, additional_params)
        
        # Handle cases where combinations don't match target
        if total_combinations < target_count:
            # Expand parameters
            if 'additional_axes' in n_config:
                balance_across = self.expand_parameters(balance_across, total_combinations, target_count)
                total_combinations = self.calculate_combinations(balance_across, additional_params)
            
            # If still insufficient, repeat combinations
            repetitions = max(1, target_count // total_combinations)
        else:
            repetitions = 1
        
        # Generate actual combinations
        if n_points == 1:
            configs.extend(self._generate_n1_configs(n_config, target_count, chart_type))
        elif n_points == 2:
            configs.extend(self._generate_n2_configs(n_config, target_count, chart_type))
        elif n_points in [3, 4]:
            configs.extend(self._generate_n3_n4_configs(n_config, n_points, target_count, chart_type))
        elif n_points in [8, 16]:
            configs.extend(self._generate_n_large_configs(n_config, n_points, target_count, chart_type))

        # Subsample if necessary
        if len(configs) > target_count:
            strategy = n_config.get('subsampling_strategy', 'random')
            indices = self.subsample_combinations(configs, target_count, strategy)
            configs = [configs[i] for i in indices]
        
        return configs
    
    def _generate_n1_configs(self, n_config: Dict, target_count: int, chart_type: str = "scatter") -> List[StimulusConfig]:
        """Generate configurations for n=1"""
        configs = []
        dot_objects = self.get_dot_objects()
        
        # Extract tick configs and point generators from balance_across
        tick_configs = [0]  # default
        point_generators = None
        for item in n_config['balance_across']:
            if isinstance(item, dict):
                if 'tick_configs' in item:
                    tick_configs = item['tick_configs']
                elif 'point_generators' in item:
                    point_generators = item['point_generators']
        
        # Set default if not found in config
        if point_generators is None:
            point_generators = ["FixedPointGenerator", "UniformPointGenerator"]
        
        # Generate balanced combinations
        balanced_combinations = []
        
        # Calculate combinations per generator to maximize balance
        n_generators = len(point_generators)
        n_tick_configs = len(tick_configs)
        combinations_per_generator = target_count // n_generators
        extra_combinations = target_count % n_generators
        
        # Create all possible dot-position-tick combinations
        dot_pos_tick_combinations = []
        for tick_config_idx in tick_configs:
            tick_config = self.parameter_spaces['tick_configs'][tick_config_idx]
            grid_positions = self.get_grid_positions(tick_config)
            for dot_obj in dot_objects:
                for pos in grid_positions:
                    dot_pos_tick_combinations.append((dot_obj, pos, tick_config_idx))
        
        # Shuffle to ensure randomness within balanced structure
        np.random.shuffle(dot_pos_tick_combinations)
        
        # Create balanced combinations
        combo_idx = 0
        for gen_idx, gen_type in enumerate(point_generators):
            # Determine how many combinations this generator should get
            gen_count = combinations_per_generator + (1 if gen_idx < extra_combinations else 0)
            
            for _ in range(gen_count):
                if combo_idx >= len(dot_pos_tick_combinations):
                    # If we run out of combinations, cycle back
                    combo_idx = 0
                
                dot_obj, pos, tick_config_idx = dot_pos_tick_combinations[combo_idx]
                balanced_combinations.append((dot_obj, pos, tick_config_idx, gen_type))
                combo_idx += 1
                
                if len(balanced_combinations) >= target_count:
                    break
            if len(balanced_combinations) >= target_count:
                break
        
        # Trim to exact target count and shuffle final list
        balanced_combinations = balanced_combinations[:target_count]
        np.random.shuffle(balanced_combinations)
        
        # Create configs
        for dot_obj, pos, tick_config_idx, gen_type in balanced_combinations:
            if gen_type == "FixedPointGenerator":
                point_gen = FixedPointGenerator(x_position=[pos[0]], y_position=[pos[1]])
            else:
                point_gen = UniformPointGenerator()
            
            tick_config = self.parameter_spaces['tick_configs'][tick_config_idx]
            config = StimulusConfig(
                n_points=1,
                axis_config='xy',
                color=[dot_obj[0]],
                dot_size=2000,
                dot_shape=[dot_obj[1]],
                axis_range=tuple(tick_config['axis_range']),
                tick_scale=tick_config['tick_scale'],
                n_ticks=tick_config['n_ticks'],
                chart_type=chart_type
            )
            configs.append((point_gen, config))
        
        return configs
    
    def _generate_n2_configs(self, n_config: Dict, target_count: int, chart_type: str = "scatter") -> List[StimulusConfig]:
        """Generate configurations for n=2"""
        configs = []
        
        # Extract tick configs and point generators from balance_across
        tick_configs = [0]  # default
        point_generators = None
        disjoint_dot_pairs_count = 4
        spatial_configs = ['x']

        for item in n_config['balance_across']:
            if isinstance(item, dict):
                if 'tick_configs' in item:
                    tick_configs = item['tick_configs']
                elif 'point_generators' in item:
                    point_generators = item['point_generators']
                elif 'disjoint_dot_pairs' in item:
                    disjoint_dot_pairs_count = item['disjoint_dot_pairs']
                elif 'spatial_configs' in item:
                    spatial_configs = item['spatial_configs']
        
        # Set default if not found in config
        if point_generators is None:
            point_generators = ["FixedPointGenerator", "UniformPointGenerator"]
        
        # Generate balanced combinations
        balanced_combinations = []
        
        # Calculate combinations per generator to maximize balance
        n_generators = len(point_generators)
        combinations_per_generator = target_count // n_generators
        extra_combinations = target_count % n_generators
        
        # Create all possible position-pair, dot-pair, and tick combinations
        pos_dot_tick_combinations = []
        for tick_config_idx in tick_configs:
            tick_config = self.parameter_spaces['tick_configs'][tick_config_idx]
            grid_positions = self.get_grid_positions(tick_config)
            position_pairs = list(itertools.combinations(grid_positions, 2))
            for pos_pair in position_pairs:
                dot_objects = self.get_dot_objects()
                dot_pairs = self.get_disjoint_dot_pairs(dot_objects, n_pairs=disjoint_dot_pairs_count)
                for dot_pair in dot_pairs:
                    for spatial_config in spatial_configs:
                        pos_dot_tick_combinations.append((pos_pair, dot_pair, tick_config_idx, spatial_config))
        
        # Shuffle to ensure randomness within balanced structure
        np.random.shuffle(pos_dot_tick_combinations)
        
        # Create balanced combinations
        combo_idx = 0
        for gen_idx, gen_type in enumerate(point_generators):
            # Determine how many combinations this generator should get
            gen_count = combinations_per_generator + (1 if gen_idx < extra_combinations else 0)
            
            for _ in range(gen_count):
                if combo_idx >= len(pos_dot_tick_combinations):
                    # If we run out of combinations, cycle back
                    combo_idx = 0
                
                pos_pair, dot_pair, tick_config_idx, spatial_config = pos_dot_tick_combinations[combo_idx]
                balanced_combinations.append((pos_pair, dot_pair, tick_config_idx, spatial_config, gen_type))
                combo_idx += 1
                
                if len(balanced_combinations) >= target_count:
                    break
            if len(balanced_combinations) >= target_count:
                break
        
        # Trim to exact target count and shuffle final list
        balanced_combinations = balanced_combinations[:target_count]
        np.random.shuffle(balanced_combinations)
        
        # Create configs
        for pos_pair, dot_pair, tick_config_idx, spatial_config, gen_type in balanced_combinations:
            if gen_type == "FixedPointGenerator":
                point_gen = FixedPointGenerator(
                    x_position=[pos_pair[0][0], pos_pair[1][0]],
                    y_position=[pos_pair[0][1], pos_pair[1][1]]
                )
            else:
                point_gen = UniformPointGenerator()
            
            tick_config = self.parameter_spaces['tick_configs'][tick_config_idx]
            config = StimulusConfig(
                n_points=2,
                axis_config='xy',
                color=[d[0] for d in dot_pair],
                dot_size=2000,
                dot_shape=[d[1] for d in dot_pair],
                axis_range=tuple(tick_config['axis_range']),
                tick_scale=tick_config['tick_scale'],
                n_ticks=tick_config['n_ticks'],
                chart_type=chart_type,
                spatial_config=spatial_config
            )
            configs.append((point_gen, config))

        return configs
    
    def _generate_n3_n4_configs(self, n_config: Dict, n_points: int, target_count: int, chart_type: str = "scatter") -> List[StimulusConfig]:
        """Generate configurations for n=3 and n=4"""
        configs = []
        dot_objects = self.get_dot_objects()
        
        # Extract parameters from balance_across
        tick_configs = [0]  # default
        point_generators = None
        spatial_configs = None
        buddy_samples_count = 10
        
        for item in n_config['balance_across']:
            if isinstance(item, dict):
                if 'tick_configs' in item:
                    tick_configs = item['tick_configs']
                elif 'point_generators' in item:
                    point_generators = item['point_generators']
                elif 'buddy_samples' in item:
                    buddy_samples_count = item['buddy_samples']
                elif 'spatial_configs' in item:
                    spatial_configs = item['spatial_configs']
        
        # Set defaults if not found in config
        if point_generators is None:
            point_generators = ["FixedPointGenerator", "UniformPointGenerator"]
        if spatial_configs is None:
            spatial_configs = ["x_collinear", "y_collinear", "2d"]
            
        collinear_positions = n_config.get('additional_params', {}).get('collinear_positions', 8)
        twod_positions = n_config.get('additional_params', {}).get('twod_positions', 8)
        
        # Generate balanced combinations
        balanced_combinations = []
        
        # Calculate how many combinations we need per category
        n_generators = len(point_generators)
        n_spatial = len(spatial_configs)
        n_dots = len(dot_objects)
        
        # Calculate combinations per category to maximize balance
        combinations_per_generator = target_count // n_generators
        extra_combinations = target_count % n_generators
        
        combinations_per_spatial = target_count // n_spatial
        extra_spatial = target_count % n_spatial
        
        # Generate dot objects and buddy samples with tick configs
        dot_buddy_tick_combinations = []
        for tick_config_idx in tick_configs:
            for dot_obj in dot_objects:
                buddy_samples = self.generate_buddy_samples(dot_objects, dot_obj, n_points-1, buddy_samples_count)
                for buddy_sample in buddy_samples:
                    dot_buddy_tick_combinations.append((buddy_sample, tick_config_idx))
        
        # Shuffle to ensure randomness within balanced structure
        np.random.shuffle(dot_buddy_tick_combinations)
        
        # Create balanced combinations
        combo_idx = 0
        for gen_idx, gen_type in enumerate(point_generators):
            # Determine how many combinations this generator should get
            gen_count = combinations_per_generator + (1 if gen_idx < extra_combinations else 0)
            
            for spatial_idx, spatial_config in enumerate(spatial_configs):
                # Determine how many combinations this spatial config should get within this generator
                spatial_count = gen_count // n_spatial + (1 if spatial_idx < (gen_count % n_spatial) else 0)
                
                for _ in range(spatial_count):
                    if combo_idx >= len(dot_buddy_tick_combinations):
                        # If we run out of combinations, cycle back
                        combo_idx = 0
                    
                    buddy_sample, tick_config_idx = dot_buddy_tick_combinations[combo_idx]
                    positions_count = collinear_positions if spatial_config in ["x_collinear", "y_collinear"] else twod_positions
                    
                    balanced_combinations.append((buddy_sample, spatial_config, tick_config_idx, gen_type))
                    combo_idx += 1
                    
                    if len(balanced_combinations) >= target_count:
                        break
                if len(balanced_combinations) >= target_count:
                    break
            if len(balanced_combinations) >= target_count:
                break
        
        # Trim to exact target count and shuffle final list
        balanced_combinations = balanced_combinations[:target_count]
        np.random.shuffle(balanced_combinations)
        
        # Create configs
        for buddy_sample, spatial_config, tick_config_idx, gen_type in balanced_combinations:
            # Get tick config and axis range
            tick_config = self.parameter_spaces['tick_configs'][tick_config_idx]
            axis_range = tick_config['axis_range']
            max_pos = axis_range[1]
            
            if gen_type == "FixedPointGenerator":
                if spatial_config == "x_collinear":
                    x = np.random.randint(1, max_pos + 1)
                    y_positions = np.random.choice(range(1, max_pos + 1), size=n_points, replace=False)
                    point_gen = FixedPointGenerator(
                        x_position=[x] * n_points,
                        y_position=y_positions.tolist()
                    )
                elif spatial_config == "y_collinear":
                    y = np.random.randint(1, max_pos + 1)
                    x_positions = np.random.choice(range(1, max_pos + 1), size=n_points, replace=False)
                    point_gen = FixedPointGenerator(
                        x_position=x_positions.tolist(),
                        y_position=[y] * n_points
                    )
                else:  # 2d
                    point_gen = GridPointGenerator()
            else:  # UniformPointGenerator
                if spatial_config == "x_collinear":
                    point_gen = UniformPointGenerator(x_position=np.random.randint(1, max_pos + 1))
                elif spatial_config == "y_collinear":
                    point_gen = UniformPointGenerator(y_position=np.random.randint(1, max_pos + 1))
                else:  # 2d
                    point_gen = UniformPointGenerator()
            
            config = StimulusConfig(
                n_points=n_points,
                axis_config='xy',
                color=[d[0] for d in buddy_sample],
                dot_size=2000,
                dot_shape=[d[1] for d in buddy_sample],
                axis_range=tuple(tick_config['axis_range']),
                tick_scale=tick_config['tick_scale'],
                n_ticks=tick_config['n_ticks'],
                chart_type=chart_type,
                spatial_config=spatial_config
            )
            configs.append((point_gen, config))
        
        return configs
    
    def _generate_n_large_configs(self, n_config: Dict, n_points: int, target_count: int, chart_type: str = "scatter") -> List[StimulusConfig]:
        """Generate configurations for n=8 and n=16"""
        configs = []
        dot_objects = self.get_dot_objects()
        
        # Extract tick configs and point generators from balance_across
        tick_configs = [0]  # default
        point_generators = None
        buddy_samples_count = 10
        spatial_configs = None

        for item in n_config['balance_across']:
            if isinstance(item, dict):
                if 'tick_configs' in item:
                    tick_configs = item['tick_configs']
                elif 'point_generators' in item:
                    point_generators = item['point_generators']
                elif 'buddy_samples' in item:
                    buddy_samples_count = item['buddy_samples']
                elif 'spatial_configs' in item:
                    spatial_configs = item['spatial_configs']

        # Set default if not found in config
        if point_generators is None:
            point_generators = ["GridPointGenerator", "UniformPointGenerator", "MeanCenteredPointGenerator"]
        if spatial_configs is None:
            spatial_configs = ['x'] if chart_type == 'bar' else ['2d']
            
        mean_params = n_config.get('additional_params', {}).get('mean_centered_params', {})
        
        # Generate balanced combinations
        balanced_combinations = []
        
        # Calculate combinations per generator to maximize balance
        n_generators = len(point_generators)
        combinations_per_generator = target_count // n_generators
        extra_combinations = target_count % n_generators
        
        # Generate dot objects and buddy samples with tick configs
        dot_buddy_tick_combinations = []
        for tick_config_idx in tick_configs:
            for dot_obj in dot_objects:
                buddy_samples = self.generate_buddy_samples(dot_objects, dot_obj, n_points-1, buddy_samples_count)
                for buddy_sample in buddy_samples:
                    dot_buddy_tick_combinations.append((buddy_sample, tick_config_idx))
        
        # Shuffle to ensure randomness within balanced structure
        np.random.shuffle(dot_buddy_tick_combinations)
        
        # Create balanced combinations
        combo_idx = 0
        for gen_idx, gen_type in enumerate(point_generators):
            # Determine how many combinations this generator should get
            gen_count = combinations_per_generator + (1 if gen_idx < extra_combinations else 0)

            for _ in range(gen_count):
                if combo_idx >= len(dot_buddy_tick_combinations):
                    # If we run out of combinations, cycle back
                    combo_idx = 0

                buddy_sample, tick_config_idx = dot_buddy_tick_combinations[combo_idx]
                spatial_config = spatial_configs[len(balanced_combinations) % len(spatial_configs)]
                balanced_combinations.append((buddy_sample, tick_config_idx, gen_type, spatial_config))
                combo_idx += 1
                
                if len(balanced_combinations) >= target_count:
                    break
            if len(balanced_combinations) >= target_count:
                break
        
        # Trim to exact target count and shuffle final list
        balanced_combinations = balanced_combinations[:target_count]
        np.random.shuffle(balanced_combinations)
        
        # Create configs
        for buddy_sample, tick_config_idx, gen_type, spatial_config in balanced_combinations:
            tick_config = self.parameter_spaces['tick_configs'][tick_config_idx]

            if gen_type == "GridPointGenerator":
                point_gen = GridPointGenerator()
            elif gen_type == "UniformPointGenerator":
                point_gen = UniformPointGenerator()
            else:  # MeanCenteredPointGenerator
                # Use axis range from tick config, or fall back to mean_params
                axis_range = tick_config['axis_range']
                max_pos = axis_range[1]
                default_range = [1, max_pos]
                
                mean_x = np.random.randint(mean_params.get('mean_x_range', default_range)[0], 
                                         mean_params.get('mean_x_range', default_range)[1] + 1)
                mean_y = np.random.randint(mean_params.get('mean_y_range', default_range)[0], 
                                         mean_params.get('mean_y_range', default_range)[1] + 1)
                point_gen = MeanCenteredPointGenerator(
                    mean=(mean_x, mean_y), 
                    std=mean_params.get('std', 0.5)
                )
            
            config = StimulusConfig(
                n_points=n_points,
                axis_config='xy',
                color=[d[0] for d in buddy_sample],
                dot_size=2000,
                dot_shape=[d[1] for d in buddy_sample],
                axis_range=tuple(tick_config['axis_range']),
                tick_scale=tick_config['tick_scale'],
                n_ticks=tick_config['n_ticks'],
                chart_type=chart_type,
                spatial_config=spatial_config
            )
            configs.append((point_gen, config))
        
        return configs
    
    def generate_ensemble_configs(self, ensemble_config: Dict) -> List[Tuple]:
        """Generate configurations for ensemble stimuli"""
        configs = []
        
        for stim_type, type_config in ensemble_config.items():
            if stim_type == 'target_count':
                continue
                
            configs.extend(self._generate_ensemble_type_configs(stim_type, type_config))
        
        return configs
    
    def _generate_ensemble_type_configs(self, stim_type: str, type_config: Dict) -> List[Tuple]:
        """Generate configurations for a specific ensemble stimulus type"""
        configs = []
        target_count = type_config['target_count']
        n_points_list = type_config['n_points']
        
        if stim_type == 'correlation':
            configs.extend(self._generate_correlation_configs(type_config, target_count))
        elif stim_type == 'function':
            configs.extend(self._generate_function_configs(type_config, target_count))
        elif stim_type == 'cluster':
            configs.extend(self._generate_cluster_configs(type_config, target_count))
        elif stim_type == 'outlier':
            configs.extend(self._generate_outlier_configs(type_config, target_count))
        
        return configs
    
    def _generate_correlation_configs(self, type_config: Dict, target_count: int) -> List[Tuple]:
        """Generate correlation stimulus configurations"""
        configs = []
        n_points_list = type_config['n_points']
        colors = self.parameter_spaces['colors']
        shapes = self.parameter_spaces['shapes']
        
        # Extract tick configs from balance_across
        tick_configs = [0]  # default
        for item in type_config['balance_across']:
            if isinstance(item, dict) and 'tick_configs' in item:
                tick_configs = item['tick_configs']
                break
        
        # Generate base combinations
        base_combinations = []
        for n in n_points_list:
            for r in self.parameter_spaces['correlation_params']['r_values']:
                # Select appropriate intercept proportions based on correlation sign
                if r > 0:
                    intercept_proportions = self.parameter_spaces['correlation_params']['positive_intercept_proportions']
                else:
                    intercept_proportions = self.parameter_spaces['correlation_params']['negative_intercept_proportions']
                
                for intercept_proportion in intercept_proportions:
                    for noise in self.parameter_spaces['correlation_params']['noise_levels']:
                        for tick_config_idx in tick_configs:
                            # Compute actual intercept based on axis range
                            tick_config = self.parameter_spaces['tick_configs'][tick_config_idx]
                            actual_intercept = tick_config['axis_range'][1] * intercept_proportion
                            
                            base_combinations.append({
                                'n': n, 'r': r, 'intercept': actual_intercept, 'noise': noise, 'tick_config_idx': tick_config_idx
                            })
        
        # Shuffle and repeat to reach target count
        np.random.shuffle(base_combinations)
        combinations_to_generate = []
        while len(combinations_to_generate) < target_count:
            combinations_to_generate.extend(base_combinations)
        combinations_to_generate = combinations_to_generate[:target_count]
        
        # Create configs
        for combo in combinations_to_generate:
            point_gen = CorrelationPointGenerator(r=combo['r'], intercept=combo['intercept'], noise=combo['noise'])
            color = np.random.choice(colors)
            shape = np.random.choice(shapes)
            tick_config = self.parameter_spaces['tick_configs'][combo['tick_config_idx']]
            
            config = StimulusConfig(
                n_points=combo['n'],
                axis_config='xy',
                color=color,
                dot_size=2000,
                dot_shape=shape,
                axis_range=tuple(tick_config['axis_range']),
                tick_scale=tick_config['tick_scale'],
                n_ticks=tick_config['n_ticks']
            )
            configs.append((point_gen, config))
        
        return configs
    
    def _generate_function_configs(self, type_config: Dict, target_count: int) -> List[Tuple]:
        """Generate function stimulus configurations"""
        configs = []
        n_points_list = type_config['n_points']
        colors = self.parameter_spaces['colors']
        shapes = self.parameter_spaces['shapes']
        
        # Extract tick configs from balance_across
        tick_configs = [0]  # default
        for item in type_config['balance_across']:
            if isinstance(item, dict) and 'tick_configs' in item:
                tick_configs = item['tick_configs']
                break
        
        # Generate base combinations
        base_combinations = []
        for n in n_points_list:
            for func_type in self.parameter_spaces['function_params']['types']:
                for noise in self.parameter_spaces['function_params']['noise_levels']:
                    for tick_config_idx in tick_configs:
                        # Sample 5 random parameter combinations for each function type
                        param_space = self.parameter_spaces['function_params'][func_type]
                        for _ in range(5):
                            func_params_sample = {}
                            tick_config = self.parameter_spaces['tick_configs'][tick_config_idx]
                            axis_max = tick_config['axis_range'][1]
                            
                            for k, v in param_space.items():
                                if k.endswith('_proportions'):
                                    # Convert proportions to actual values
                                    param_name = k.replace('_proportions', '')
                                    actual_value = np.random.choice(v) * axis_max
                                    func_params_sample[param_name] = actual_value
                                elif k.endswith('_values'):
                                    # Use values directly
                                    param_name = k.replace('_values', '')
                                    func_params_sample[param_name] = np.random.choice(v)
                                else:
                                    # Use values directly (slopes, a_values, etc.)
                                    func_params_sample[k] = np.random.choice(v)
                            
                            base_combinations.append({
                                'n': n,
                                'func_type': func_type,
                                'noise': noise,
                                'func_params': func_params_sample,
                                'tick_config_idx': tick_config_idx
                            })
        
        # Shuffle and repeat to reach target count
        np.random.shuffle(base_combinations)
        combinations_to_generate = []
        while len(combinations_to_generate) < target_count:
            combinations_to_generate.extend(base_combinations)
        combinations_to_generate = combinations_to_generate[:target_count]
        
        # Create configs
        for combo in combinations_to_generate:
            point_gen = FunctionPointGenerator(
                func_type=combo['func_type'],
                noise=combo['noise'],
                **combo['func_params']
            )
            color = np.random.choice(colors)
            shape = np.random.choice(shapes)
            tick_config = self.parameter_spaces['tick_configs'][combo['tick_config_idx']]
            
            config = StimulusConfig(
                n_points=combo['n'],
                axis_config='xy',
                color=color,
                dot_size=2000,
                dot_shape=shape,
                axis_range=tuple(tick_config['axis_range']),
                tick_scale=tick_config['tick_scale'],
                n_ticks=tick_config['n_ticks']
            )
            configs.append((point_gen, config))
        
        return configs
    
    def _generate_cluster_configs(self, type_config: Dict, target_count: int) -> List[Tuple]:
        """Generate cluster stimulus configurations"""
        configs = []
        n_points_list = type_config['n_points']
        colors = self.parameter_spaces['colors']
        shapes = self.parameter_spaces['shapes']
        samples_per_combination = type_config.get('samples_per_combination', 5)
        
        # Extract tick configs from balance_across
        tick_configs = [0]  # default
        for item in type_config['balance_across']:
            if isinstance(item, dict) and 'tick_configs' in item:
                tick_configs = item['tick_configs']
                break
        
        # Generate base combinations
        base_combinations = []
        for n in n_points_list:
            for cluster_std in self.parameter_spaces['cluster_params']['cluster_stds']:
                for query_cluster in self.parameter_spaces['cluster_params']['query_clusters']:
                    for ambiguity in self.parameter_spaces['cluster_params']['ambiguity_levels']:
                        for tick_config_idx in tick_configs:
                            base_combinations.append({
                                'n': n,
                                'cluster_std': cluster_std,
                                'query_cluster': query_cluster,
                                'ambiguity': ambiguity,
                                'tick_config_idx': tick_config_idx
                            })
        
        # Calculate repetitions needed
        total_base_samples = len(base_combinations) * samples_per_combination
        repetitions_needed = max(1, target_count // total_base_samples)
        extra_samples_needed = target_count % total_base_samples
        
        # Generate configs
        combinations_generated = 0
        for rep in range(repetitions_needed):
            for combo in base_combinations:
                if combinations_generated >= target_count:
                    break
                    
                for _ in range(samples_per_combination):
                    if combinations_generated >= target_count:
                        break
                        
                    point_gen = ClusterPointGenerator(
                        cluster_stds=combo['cluster_std'],
                        query_cluster=combo['query_cluster'],
                        ambiguity=combo['ambiguity']
                    )
                    
                    # Select distinct colors and shapes
                    cluster1_color = np.random.choice(colors)
                    remaining_colors = [c for c in colors if c != cluster1_color]
                    cluster2_color = np.random.choice(remaining_colors)
                    remaining_colors = [c for c in remaining_colors if c != cluster2_color]
                    query_color = np.random.choice(remaining_colors)
                    
                    cluster1_shape = np.random.choice(shapes)
                    remaining_shapes = [s for s in shapes if s != cluster1_shape]
                    cluster2_shape = np.random.choice(remaining_shapes)
                    remaining_shapes = [s for s in remaining_shapes if s != cluster2_shape]
                    query_shape = np.random.choice(remaining_shapes)
                    
                    n_per_cluster = (combo['n'] - 1) // 2
                    tick_config = self.parameter_spaces['tick_configs'][combo['tick_config_idx']]
                    
                    config = StimulusConfig(
                        n_points=combo['n'],
                        axis_config='xy',
                        color=[cluster1_color] * n_per_cluster + [cluster2_color] * n_per_cluster + [query_color],
                        dot_size=300,
                        dot_shape=[cluster1_shape] * n_per_cluster + [cluster2_shape] * n_per_cluster + [query_shape],
                        axis_range=tuple(tick_config['axis_range']),
                        tick_scale=tick_config['tick_scale'],
                        n_ticks=tick_config['n_ticks']
                    )
                    configs.append((point_gen, config))
                    combinations_generated += 1
        
        return configs
    
    def _generate_outlier_configs(self, type_config: Dict, target_count: int) -> List[Tuple]:
        """Generate outlier stimulus configurations"""
        configs = []
        n_points_list = type_config['n_points']
        colors = self.parameter_spaces['colors']
        shapes = self.parameter_spaces['shapes']
        
        # Get base distributions and tick configs from configuration
        base_distributions = None
        tick_configs = [0]  # default
        for item in type_config['balance_across']:
            if isinstance(item, dict):
                if 'base_distributions' in item:
                    base_distributions = item['base_distributions']
                elif 'tick_configs' in item:
                    tick_configs = item['tick_configs']
        
        # Set default if not found in config
        if base_distributions is None:
            base_distributions = ['correlation', 'function', 'cluster']
        
        # Create base distribution configs only for specified types
        base_configs = {dist_type: [] for dist_type in base_distributions}
        
        # Add correlation configurations if specified
        if 'correlation' in base_distributions:
            for n in n_points_list:
                for r in self.parameter_spaces['correlation_params']['r_values']:
                    # Select appropriate intercept proportions based on correlation sign
                    if r > 0:
                        intercept_proportions = self.parameter_spaces['correlation_params']['positive_intercept_proportions']
                    else:
                        intercept_proportions = self.parameter_spaces['correlation_params']['negative_intercept_proportions']
                    
                    for intercept_proportion in intercept_proportions:
                        for noise in self.parameter_spaces['correlation_params']['noise_levels']:
                            for tick_config_idx in tick_configs:
                                # Compute actual intercept based on axis range
                                tick_config = self.parameter_spaces['tick_configs'][tick_config_idx]
                                actual_intercept = tick_config['axis_range'][1] * intercept_proportion
                                
                                base_configs['correlation'].append({
                                    'type': 'correlation',
                                    'n': n,
                                    'tick_config_idx': tick_config_idx,
                                    'params': {'r': r, 'intercept': actual_intercept, 'noise': noise}
                                })
        
        # Add function configurations if specified
        if 'function' in base_distributions:
            for n in n_points_list:
                for func_type in self.parameter_spaces['function_params']['types']:
                    for noise in self.parameter_spaces['function_params']['noise_levels']:
                        for tick_config_idx in tick_configs:
                            param_space = self.parameter_spaces['function_params'][func_type]
                            tick_config = self.parameter_spaces['tick_configs'][tick_config_idx]
                            axis_max = tick_config['axis_range'][1]
                            
                            func_params_sample = {}
                            for k, v in param_space.items():
                                if k.endswith('_proportions'):
                                    # Convert proportions to actual values
                                    param_name = k.replace('_proportions', '')
                                    actual_value = np.random.choice(v) * axis_max
                                    func_params_sample[param_name] = actual_value
                                elif k.endswith('_values'):
                                    # Use values directly
                                    param_name = k.replace('_values', '')
                                    func_params_sample[param_name] = np.random.choice(v)
                                else:
                                    # Use values directly (slopes, a_values, etc.)
                                    func_params_sample[k] = np.random.choice(v)
                            
                            base_configs['function'].append({
                                'type': 'function',
                                'n': n,
                                'tick_config_idx': tick_config_idx,
                                'params': {
                                    'func_type': func_type,
                                    'noise': noise,
                                    **func_params_sample
                                }
                            })
        
        # Add cluster configurations if specified
        if 'cluster' in base_distributions:
            for n in n_points_list:
                for cluster_std in self.parameter_spaces['cluster_params']['cluster_stds']:
                    for query_cluster in self.parameter_spaces['cluster_params']['query_clusters']:
                        for ambiguity in self.parameter_spaces['cluster_params']['ambiguity_levels']:
                            for tick_config_idx in tick_configs:
                                base_configs['cluster'].append({
                                    'type': 'cluster',
                                    'n': n,
                                    'tick_config_idx': tick_config_idx,
                                    'params': {
                                        'cluster_stds': cluster_std,
                                        'query_cluster': query_cluster,
                                        'ambiguity': ambiguity
                                    }
                                })
        
        # Calculate how many base configs we need per base distribution type
        # We'll multiply by std_multipliers later, so divide by that count
        total_multipliers = len(self.parameter_spaces['outlier_params']['std_multipliers'])
        base_configs_needed = target_count // total_multipliers
        configs_per_base_type = base_configs_needed // len(base_distributions)
        extra_configs = base_configs_needed % len(base_distributions)
        
        sampled_configs = []
        for i, stim_type in enumerate(base_distributions):
            if stim_type in base_configs:
                type_configs = base_configs[stim_type]
                np.random.shuffle(type_configs)
                # Distribute extra configs to first few types
                num_configs = configs_per_base_type + (1 if i < extra_configs else 0)
                sampled_configs.extend(type_configs[:num_configs])
        
        # Generate outlier configs
        for config in sampled_configs:
            for std_multiplier in self.parameter_spaces['outlier_params']['std_multipliers']:
                if len(configs) >= target_count:
                    break
                    
                # Create appropriate base generator
                if config['type'] == 'correlation':
                    base_gen = CorrelationPointGenerator(**config['params'])
                elif config['type'] == 'function':
                    base_gen = FunctionPointGenerator(**config['params'])
                else:  # cluster
                    # Use single cluster generator for outlier base distribution
                    # Randomly select from single cluster stds for variety
                    if 'single_cluster_stds' in self.parameter_spaces['outlier_params']:
                        cluster_std = np.random.choice(self.parameter_spaces['outlier_params']['single_cluster_stds'])
                    else:
                        cluster_std = config['params']['cluster_stds'][0]  # Fallback to first cluster std
                    base_gen = SingleClusterPointGenerator(cluster_std=cluster_std)
                
                point_gen = OutlierPointGenerator(base_gen, std_multiplier=std_multiplier)
                
                tick_config = self.parameter_spaces['tick_configs'][config['tick_config_idx']]
                
                if config['type'] == 'cluster':
                    # Same color/shape for cluster and outlier (outlier should match base cluster)
                    cluster_color = np.random.choice(colors)
                    cluster_shape = np.random.choice(shapes)
                    
                    stim_config = StimulusConfig(
                        n_points=config['n'],
                        axis_config='xy',
                        color=[cluster_color] * config['n'],  # All points same color
                        dot_size=300,
                        dot_shape=[cluster_shape] * config['n'],  # All points same shape
                        axis_range=tuple(tick_config['axis_range']),
                        tick_scale=tick_config['tick_scale'],
                        n_ticks=tick_config['n_ticks']
                    )
                else:
                    # Same color/shape for all points
                    color = np.random.choice(colors)
                    shape = np.random.choice(shapes)
                    
                    stim_config = StimulusConfig(
                        n_points=config['n'],
                        axis_config='xy',
                        color=color,
                        dot_size=300,
                        dot_shape=shape,
                        axis_range=tuple(tick_config['axis_range']),
                        tick_scale=tick_config['tick_scale'],
                        n_ticks=tick_config['n_ticks']
                    )
                
                configs.append((point_gen, stim_config))
            
            if len(configs) >= target_count:
                break
        
        # If we still haven't reached target count, repeat configurations
        while len(configs) < target_count:
            original_count = len(configs)
            for config in sampled_configs:
                for std_multiplier in self.parameter_spaces['outlier_params']['std_multipliers']:
                    if len(configs) >= target_count:
                        break
                        
                    # Create appropriate base generator
                    if config['type'] == 'correlation':
                        base_gen = CorrelationPointGenerator(**config['params'])
                    elif config['type'] == 'function':
                        base_gen = FunctionPointGenerator(**config['params'])
                    else:  # cluster
                        # Use single cluster generator for outlier base distribution
                        # Randomly select from single cluster stds for variety
                        if 'single_cluster_stds' in self.parameter_spaces['outlier_params']:
                            cluster_std = np.random.choice(self.parameter_spaces['outlier_params']['single_cluster_stds'])
                        else:
                            cluster_std = config['params']['cluster_stds'][0]  # Fallback to first cluster std
                        base_gen = SingleClusterPointGenerator(cluster_std=cluster_std)
                    
                    point_gen = OutlierPointGenerator(base_gen, std_multiplier=std_multiplier)
                    
                    tick_config = self.parameter_spaces['tick_configs'][config['tick_config_idx']]
                    
                    if config['type'] == 'cluster':
                        # Same color/shape for cluster and outlier (outlier should match base cluster)
                        cluster_color = np.random.choice(colors)
                        cluster_shape = np.random.choice(shapes)
                        
                        stim_config = StimulusConfig(
                            n_points=config['n'],
                            axis_config='xy',
                            color=[cluster_color] * config['n'],  # All points same color
                            dot_size=300,
                            dot_shape=[cluster_shape] * config['n'],  # All points same shape
                            axis_range=tuple(tick_config['axis_range']),
                            tick_scale=tick_config['tick_scale'],
                            n_ticks=tick_config['n_ticks']
                        )
                    else:
                        # Same color/shape for all points
                        color = np.random.choice(colors)
                        shape = np.random.choice(shapes)
                        
                        stim_config = StimulusConfig(
                            n_points=config['n'],
                            axis_config='xy',
                            color=color,
                            dot_size=300,
                            dot_shape=shape,
                            axis_range=tuple(tick_config['axis_range']),
                            tick_scale=tick_config['tick_scale'],
                            n_ticks=tick_config['n_ticks']
                        )
                    
                    configs.append((point_gen, stim_config))
                
                if len(configs) >= target_count:
                    break
            
            # Safety check to avoid infinite loop
            if len(configs) == original_count:
                break
        
        return configs[:target_count]
    
    def generate_dataset(self, dataset_name: str = 'new_dataset') -> None:
        """Generate a complete dataset based on configuration"""
        dataset_config = self.config['datasets'][dataset_name]
        global_config = self.config['global']
        
        # Include dataset name in root directory for separation
        base_root_dir = global_config['root_dir']
        root_dir = f"{base_root_dir}/{dataset_name}" if dataset_name != 'main' else f"{base_root_dir}"
        generate_segmentation = global_config['generate_segmentation']
        
        total_stimuli = dataset_config['total_stimuli']
        progress_bar = tqdm.tqdm(total=total_stimuli, desc=f"Generating {dataset_name} dataset")
        
        # Generate main stimuli
        if 'main_stimuli' in dataset_config:
            main_config = dataset_config['main_stimuli']
            
            for n_key, n_config in main_config.items():
                if n_key == 'target_count':
                    continue
                    
                n_points = int(n_key.split('_')[1])
                configs = self.generate_stimulus_configs_for_n(n_config, n_points)
                
                for point_gen, stim_config in configs:
                    stimulus_gen = StimulusGenerator(
                        point_gen, 
                        root_dir=root_dir, 
                        generate_segmentation=generate_segmentation
                    )
                    stimulus_gen.generate_dataset([stim_config], n_samples=1)
                    progress_bar.update(1)
        
        # Generate ensemble stimuli
        if 'ensemble_stimuli' in dataset_config:
            ensemble_config = dataset_config['ensemble_stimuli']
            configs = self.generate_ensemble_configs(ensemble_config)
            
            for point_gen, stim_config in configs:
                stimulus_gen = StimulusGenerator(
                    point_gen, 
                    root_dir=root_dir, 
                    generate_segmentation=generate_segmentation
                )
                stimulus_gen.generate_dataset([stim_config], n_samples=1)
                progress_bar.update(1)
        
        progress_bar.close()

def generate_dot_objects():
    """Generate all possible color-shape combinations."""
    colors = ["red", "blue", "green", "orange"]
    shapes = ["o", "^", "*", "s"]
    return list(itertools.product(colors, shapes))

def get_disjoint_dot_pairs(dot_objects, n_pairs=1):
    """Generate disjoint pairs of dot objects."""
    all_pairs = list(itertools.combinations(dot_objects, 2))
    np.random.shuffle(all_pairs)
    
    # Take pairs until we have enough disjoint ones
    disjoint_pairs = []
    used_dots = set()
    
    for pair in all_pairs:
        if len(disjoint_pairs) == n_pairs:
            break
        if not (pair[0] in used_dots or pair[1] in used_dots):
            disjoint_pairs.append(pair)
            used_dots.add(pair[0])
            used_dots.add(pair[1])
    
    return disjoint_pairs

def generate_buddy_samples(dot_objects, main_dot, n_buddies, n_samples=10):
    """Generate multiple samples of buddy dots for a main dot."""
    other_dots = [d for d in dot_objects if d != main_dot]
    buddy_samples = []
    for _ in range(n_samples):
        buddies = list(np.random.choice(range(len(other_dots)), size=n_buddies, replace=False))
        buddy_samples.append([main_dot] + [other_dots[i] for i in buddies])
    return buddy_samples

if __name__ == "__main__":
    # Add argument parsing for different modes
    import argparse
    parser = argparse.ArgumentParser(description='Generate scatter plot stimuli from configuration files')
    parser.add_argument('--mode', choices=['generate', 'segment'], default='generate',
                      help='Mode to run in: generate new stimuli, generate segmentation for existing stimuli')
    parser.add_argument('--metadata-file', type=str, help='Metadata file to generate segmentation for')
    parser.add_argument('--no-segmentation', action='store_true', help='Skip segmentation generation for new stimuli')
    parser.add_argument('--config', type=str, default='datasets/bar_chart.yaml', 
                      help='Path to configuration file (default: datasets/bar_chart.yaml)')
    parser.add_argument('--dataset', type=str, default='bar_charts',
                      help='Dataset name in config file to generate (default: bar_charts)')
    args = parser.parse_args()
    
    if args.mode == 'generate':
        generator = ConfigurableDatasetGenerator(args.config)
        generator.generate_dataset(args.dataset)
    elif args.mode == 'segment':
        if args.metadata_file is None:
            raise ValueError("Must provide --metadata-file when using --mode segment")
        generate_segmentation_for_metadata(args.metadata_file)