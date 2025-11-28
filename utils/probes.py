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

        if config.chart_type == "line" and getattr(config, "use_custom_colors", False):
            config.color = config.plot_color
            return

        super()._generate_chart_colors(config)

    def _assign_line_shapes(self, config: StimulusConfig) -> None:  # type: ignore[override]
        if config.chart_type == "line" and getattr(config, "use_custom_shapes", False):
            return

        super()._assign_line_shapes(config)


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
    n_samples_per_value: int = 50,
    n_points_range: Tuple[int, int] = (1, 8),
    axis_range: Tuple[int, int] = (0, 8),
    tick_scale: int | None = None,
    base_output_dir: str = "datasets/probes/bar",
) -> None:
    """Generate a bar-chart probe dataset for a specific target color.

    Each generated chart contains exactly one bar with ``target_color`` whose
    height cycles through the provided ``target_values``. For every target
    value, ``n_samples_per_value`` stimuli are produced with the number of bars
    (``n_points``) sampled uniformly from ``n_points_range``.
    """

    dataset_root = os.path.join(base_output_dir, target_color)
    _ensure_output_dir(dataset_root)

    tick_scale = tick_scale if tick_scale is not None else axis_range[1] + 1

    min_points, max_points = n_points_range

    for target_value in target_values:
        for _ in range(n_samples_per_value):
            n_points = int(np.random.randint(min_points, max_points + 1))
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


def create_line_probe_dataset(
    target_color: str,
    target_shape: str,
    *,
    n_samples_per_position: int = 10,
    n_points_range: Tuple[int, int] = (2, 8),
    axis_range: Tuple[int, int] = (0, 8),
    tick_scale: int | None = None,
    base_output_dir: str = "datasets/probes/line",
) -> None:
    """Generate a line-chart probe dataset for a specific marker identity."""

    dataset_root = os.path.join(base_output_dir, f"{target_color}_{target_shape}")
    _ensure_output_dir(dataset_root)

    tick_scale = tick_scale if tick_scale is not None else axis_range[1] + 1
    min_points, max_points = n_points_range

    palette = ["red", "blue", "green", "orange", "purple", "brown", "cyan", "magenta"]
    shape_lookup = {"circle": "o", "square": "s", "triangle": "^", "star": "*"}
    if target_shape not in shape_lookup:
        raise ValueError(f"Unsupported target shape: {target_shape}")

    target_shape_symbol = shape_lookup[target_shape]
    other_shape_symbols = [symbol for shape, symbol in shape_lookup.items() if shape != target_shape]
    other_colors = [color for color in palette if color != target_color]

    target_range = range(axis_range[0] + 1, axis_range[1] + 1)
    for target_x in target_range:
        for target_y in target_range:
            for _ in range(n_samples_per_position):
                n_points = int(np.random.randint(min_points, max_points + 1))
                available_x = [val for val in target_range if val != target_x]
                other_x = list(np.random.choice(available_x, size=n_points - 1, replace=False))
                other_y = list(np.random.choice(list(target_range), size=n_points - 1, replace=True))

                x_positions = [target_x] + other_x
                y_positions = [target_y] + other_y

                colors = [target_color] + list(np.random.choice(other_colors, size=n_points - 1, replace=True))
                shapes = [target_shape_symbol] + list(np.random.choice(other_shape_symbols, size=n_points - 1, replace=True))

                point_gen = FixedPointGenerator(x_position=x_positions, y_position=y_positions)
                stim_config = StimulusConfig(
                    n_points=n_points,
                    axis_config="xy",
                    color=colors,
                    dot_size=300,
                    dot_shape=shapes,
                    chart_type="line",
                    axis_range=axis_range,
                    tick_scale=tick_scale,
                    n_ticks=min(axis_range[1], 8),
                    spatial_config="x",
                )

                stim_config.plot_color = colors
                stim_config.plot_dot_shape = shapes
                stim_config.use_custom_colors = True
                stim_config.use_custom_shapes = True

                generator = ProbeStimulusGenerator(
                    point_gen,
                    root_dir=dataset_root,
                    generate_segmentation=True,
                )
                generator.generate_dataset([stim_config], n_samples=1)
