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
