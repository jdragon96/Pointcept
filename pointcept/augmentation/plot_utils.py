import json
import random
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

FIGURE_DIR = Path(__file__).parent / "figures"
DEFAULT_SCAN = (
    Path(__file__).resolve().parents[3] / "data/open/teeth3ds/raw/data_part_3/upper/013ZD6EX/013ZD6EX_upper.obj"
)
NUM_PLOT_POINTS = 30000

GINGIVA = "#c9c8c2"
SIDE_ONE = "#2a78d6"
SIDE_TWO = "#eb6834"
REMOVED = "#efeeea"
TEXT = "#52514e"

plt.rcParams.update({
    "font.family": "Noto Sans CJK KR",
    "axes.titlesize": 11,
    "axes.titlecolor": "#0b0b0b",
    "axes.edgecolor": "#d6d5d0",
    "xtick.color": TEXT,
    "ytick.color": TEXT,
    "figure.facecolor": "white",
})


def load_teeth3ds(obj_path):
    obj_path = Path(obj_path)
    vertices = [line.split()[1:4] for line in open(obj_path) if line.startswith("v ")]
    fdi = np.asarray(json.load(open(obj_path.with_suffix(".json")))["labels"])
    quadrant, tooth = fdi // 10, fdi % 10
    on_second_side = (quadrant == 2) | (quadrant == 4)
    segment = np.where(fdi == 0, 0, tooth + 8 * on_second_side)
    return dict(coord=np.asarray(vertices, dtype=np.float32), segment=segment)


def load_example_scan():
    random.seed(0)
    np.random.seed(0)
    obj_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SCAN
    return subsample(load_teeth3ds(obj_path), NUM_PLOT_POINTS)


def subsample(data_dict, num_points):
    index = np.random.choice(len(data_dict["coord"]), num_points, replace=False)
    return {key: value[index] for key, value in data_dict.items()}


def copy_of(data_dict):
    return {key: value.copy() for key, value in data_dict.items()}


def removed_points(before, after):
    kept = {tuple(point) for point in after["coord"]}
    return np.asarray([point for point in before["coord"] if tuple(point) not in kept]).reshape(-1, 3)


def colors_of(segment):
    colors = np.full(len(segment), GINGIVA, dtype=object)
    colors[(segment >= 1) & (segment <= 8)] = SIDE_ONE
    colors[segment >= 9] = SIDE_TWO
    return colors


def draw_top_view(ax, data_dict, title, removed=None, size=0.4, show_axes=False):
    if removed is not None:
        ax.scatter(removed[:, 0], removed[:, 1], s=size, c=REMOVED, linewidths=0)
    coord = data_dict["coord"]
    ax.scatter(coord[:, 0], coord[:, 1], s=size, c=colors_of(data_dict["segment"]), linewidths=0)
    ax.set_title(title)
    ax.set_aspect("equal")
    if show_axes:
        ax.axhline(0, color="#d6d5d0", lw=0.8, zorder=0)
        ax.axvline(0, color="#d6d5d0", lw=0.8, zorder=0)
        ax.tick_params(labelsize=8, length=2)
    else:
        ax.axis("off")


def draw_3d_view(ax, data_dict, title):
    coord = data_dict["coord"]
    ax.scatter(coord[:, 0], coord[:, 1], coord[:, 2], s=0.3, c=colors_of(data_dict["segment"]), linewidths=0)
    ax.text2D(0.5, 1.0, title, transform=ax.transAxes, ha="center", fontsize=11)
    ax.set_box_aspect(np.ptp(coord, axis=0))
    ax.view_init(elev=25, azim=-60)
    ax.set_axis_off()


def add_legend(fig, with_removed=False):
    labels = [("잇몸 (0)", GINGIVA), ("치아 1–8", SIDE_ONE), ("치아 9–16", SIDE_TWO)]
    if with_removed:
        labels.append(("제거된 점", REMOVED))
    handles = [Line2D([], [], marker="o", ls="", color=color, markersize=7) for _, color in labels]
    fig.legend(handles, [name for name, _ in labels], loc="lower center", ncol=len(labels),
               frameon=False, labelcolor=TEXT, fontsize=9)


def save_figure(fig, source_file):
    FIGURE_DIR.mkdir(exist_ok=True)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    path = FIGURE_DIR / f"{Path(source_file).stem}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"saved {path}")
