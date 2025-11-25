import numpy as np

from utils import stimuli


def test_bar_chart_numeric_axis_ticks_within_bounds(monkeypatch):
    """Ensure bar charts render numeric ticks when segmentation is disabled."""

    plotter = stimuli.StimulusPlotter()
    captured_figs = []

    # Keep the matplotlib figure available for inspection instead of saving to disk
    monkeypatch.setattr(
        plotter,
        "_format_figure",
        lambda fig, x_offset=0, y_offset=0: (captured_figs.append(fig) or fig),
    )

    config = stimuli.StimulusConfig(
        n_points=3,
        axis_config="xy",
        color="red",
        dot_size=300,
        dot_shape="o",
        chart_type="bar",
        axis_range=(0, 6),
        tick_scale=6,
        n_ticks=6,
        spatial_config="x",
    )
    config.plot_color = ["red"] * config.n_points

    points = np.column_stack((np.arange(1, config.n_points + 1), np.arange(1, config.n_points + 1)))

    plotter.plot_stimulus_without_segmentation(points, config)

    ax = captured_figs[0].axes[0]
    y_axis_labels = [text for text in ax.texts if text.get_ha() == "right"]

    assert y_axis_labels, "Expected numeric y-axis tick labels to be drawn for bar charts."

    x_min, x_max = ax.get_xlim()
    label_positions = [text.get_position()[0] for text in y_axis_labels]

    assert all(x_min <= pos <= x_max for pos in label_positions), "Numeric tick labels should be visible within axis bounds."


def test_bar_chart_y_spatial_duplicate_extrema_resolved(tmp_path):
    """Bar charts with y-oriented categories should avoid duplicate extrema."""

    config = stimuli.StimulusConfig(
        n_points=3,
        axis_config="xy",
        color="red",
        dot_size=300,
        dot_shape="o",
        chart_type="bar",
        axis_range=(0, 6),
        tick_scale=6,
        n_ticks=6,
        spatial_config="y",
    )
    point_gen = stimuli.FixedPointGenerator(x_position=[1, 1, 6], y_position=[1, 2, 3])

    generator = stimuli.StimulusGenerator(
        point_generator=point_gen,
        generate_segmentation=False,
        save_files=False,
        root_dir=str(tmp_path),
    )

    points = point_gen.generate_points(config)
    prepared_points = generator._prepare_bar_points(points, config)
    validated_points = generator._validate_chart_constraints(prepared_points, config)

    numeric_axis = 0  # x-axis holds numeric values when spatial_config == "y"
    numeric_values = validated_points[:, numeric_axis]

    assert np.sum(numeric_values == numeric_values.min()) == 1
    assert np.sum(numeric_values == numeric_values.max()) == 1


def test_bar_chart_values_integer_and_unique_extrema(tmp_path):
    """Bar chart numeric values should snap to integers with single extrema."""

    config = stimuli.StimulusConfig(
        n_points=4,
        axis_config="xy",
        color="red",
        dot_size=300,
        dot_shape="o",
        chart_type="bar",
        axis_range=(0, 8),
        tick_scale=8,
        n_ticks=8,
        spatial_config="x",
    )

    point_gen = stimuli.UniformPointGenerator()
    generator = stimuli.StimulusGenerator(
        point_generator=point_gen,
        generate_segmentation=False,
        save_files=False,
        root_dir=str(tmp_path),
    )

    points = point_gen.generate_points(config)
    prepared_points = generator._prepare_bar_points(points, config)
    validated_points = generator._validate_chart_constraints(prepared_points, config)

    numeric_axis = 1  # y-axis holds numeric values when spatial_config == "x"
    numeric_values = validated_points[:, numeric_axis]

    assert np.all(numeric_values == numeric_values.astype(int))
    assert np.sum(numeric_values == numeric_values.min()) == 1
    assert np.sum(numeric_values == numeric_values.max()) == 1
