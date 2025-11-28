import os
from typing import List, Sequence, Tuple

import numpy as np

from utils.stimuli import FixedPointGenerator, StimulusConfig, StimulusGenerator


class ProbeStimulusGenerator(StimulusGenerator):
    """Stimulus generator that can force a target attribute for probes."""

    def __init__(self, *args, target_color: str | None = None, target_color_index: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self.target_color = target_color
        self.target_color_index = target_color_index

    def _generate_chart_colors(self, config: StimulusConfig) -> None:  # type: ignore[override]
        """Force a specific color onto a designated bar for probe datasets."""
        if config.chart_type == "bar" and self.target_color is not None:
            palette = self.base_colors + self.additional_colors
            if self.target_color not in palette:
                raise ValueError(f"Target color {self.target_color} is not in the available palette: {palette}")

            other_colors = [color for color in palette if color != self.target_color]
            if config.n_points - 1 > len(other_colors):
                raise ValueError("Bar charts require a unique color per bar; not enough colors to allocate")

            sampled_other_colors = list(np.random.choice(other_colors, size=config.n_points - 1, replace=False))
            colors = sampled_other_colors
            insertion_index = max(0, min(self.target_color_index, config.n_points - 1))
            colors.insert(insertion_index, self.target_color)

            config.plot_color = colors
            config.color = colors
            return

        super()._generate_chart_colors(config)


def _ensure_output_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _build_bar_values(target_value: int, n_points: int, axis_range: Tuple[int, int]) -> List[int]:
    """Create a list of bar values with a specific target value included."""
    min_value, max_value = axis_range
    candidate_values = [val for val in range(min_value + 1, max_value + 1) if val != target_value]
    if len(candidate_values) < n_points - 1:
        raise ValueError("Not enough distinct values available to populate non-target bars")

    other_values = list(np.random.choice(candidate_values, size=n_points - 1, replace=False))
    return [target_value] + other_values


def create_bar_probe_dataset(
    target_color: str,
    target_values: Sequence[int],
    *,
    n_points: int = 4,
    axis_range: Tuple[int, int] = (0, 8),
    tick_scale: int | None = None,
    base_output_dir: str = "datasets/probes/bar",
) -> None:
    """Generate a bar-chart probe dataset for a specific target color.

    Each generated chart contains exactly one bar with ``target_color`` whose
    height cycles through the provided ``target_values``.
    """

    dataset_root = os.path.join(base_output_dir, target_color)
    _ensure_output_dir(dataset_root)

    tick_scale = tick_scale if tick_scale is not None else axis_range[1] + 1

    for target_value in target_values:
        bar_values = _build_bar_values(target_value, n_points, axis_range)

        point_gen = FixedPointGenerator(
            x_position=list(range(1, n_points + 1)),
            y_position=bar_values,
        )

        stim_config = StimulusConfig(
            n_points=n_points,
            axis_config="xy",
            color=target_color,
            dot_size=300,
            dot_shape="s",
            chart_type="bar",
            axis_range=axis_range,
            tick_scale=tick_scale,
            n_ticks=min(axis_range[1], 8),
            spatial_config="x",
        )

        generator = ProbeStimulusGenerator(
            point_gen,
            root_dir=dataset_root,
            generate_segmentation=True,
            target_color=target_color,
            target_color_index=0,
        )
        generator.generate_dataset([stim_config], n_samples=1)


def _build_line_points(
    target_x: int,
    target_y: int,
    n_points: int,
    axis_range: Tuple[int, int],
) -> Tuple[List[int], List[int]]:
    """Construct x/y coordinates for a line chart with a fixed x-position point."""
    min_value, max_value = axis_range
    available_x = [val for val in range(min_value + 1, max_value + 1) if val != target_x]
    if len(available_x) < n_points - 1:
        raise ValueError("Not enough distinct x positions to populate non-target line points")

    other_x = list(np.random.choice(available_x, size=n_points - 1, replace=False))
    all_x = [target_x] + other_x

    available_y = list(range(min_value + 1, max_value + 1))
    other_y = list(np.random.choice(available_y, size=n_points - 1, replace=True))
    all_y = [target_y] + other_y

    return all_x, all_y


def create_line_probe_dataset(
    target_x: int,
    target_y_values: Sequence[int],
    *,
    n_points: int = 4,
    axis_range: Tuple[int, int] = (0, 8),
    tick_scale: int | None = None,
    base_output_dir: str = "datasets/probes/line",
) -> None:
    """Generate a line-chart probe dataset anchored at a fixed x-value.

    Each generated chart contains a point located at ``target_x`` whose y-value
    is drawn from ``target_y_values``. The remaining points vary across charts.
    """

    dataset_root = os.path.join(base_output_dir, f"x_{target_x}")
    _ensure_output_dir(dataset_root)

    tick_scale = tick_scale if tick_scale is not None else axis_range[1] + 1

    for target_y in target_y_values:
        x_positions, y_positions = _build_line_points(target_x, target_y, n_points, axis_range)

        point_gen = FixedPointGenerator(x_position=x_positions, y_position=y_positions)
        stim_config = StimulusConfig(
            n_points=n_points,
            axis_config="xy",
            color="blue",  # Color is shared across the line; set deterministically for consistency
            dot_size=300,
            dot_shape="o",
            chart_type="line",
            axis_range=axis_range,
            tick_scale=tick_scale,
            n_ticks=min(axis_range[1], 8),
            spatial_config="x",
        )

        generator = ProbeStimulusGenerator(
            point_gen,
            root_dir=dataset_root,
            generate_segmentation=True,
        )
        generator.generate_dataset([stim_config], n_samples=1)
