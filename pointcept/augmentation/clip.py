import numpy as np

from pointcept.datasets.transform import TRANSFORMS, index_operator


@TRANSFORMS.register_module()
class BoxClip(object):
    def __init__(self, box=(-1, -1, -1, 1, 1, 1)):
        self.box_min = np.asarray(box[:3])
        self.box_max = np.asarray(box[3:])

    def __call__(self, data_dict):
        if "coord" not in data_dict.keys():
            return data_dict
        coord = data_dict["coord"]
        inside = np.all((coord >= self.box_min) & (coord <= self.box_max), axis=1)
        return index_operator(data_dict, inside)


def visualize():
    from matplotlib.patches import Rectangle
    from plot_utils import add_legend, copy_of, draw_top_view, load_example_scan, plt, removed_points, save_figure

    scan = load_example_scan()
    centered = scan["coord"] - scan["coord"].mean(axis=0)
    scan["coord"] = centered / np.linalg.norm(centered, axis=1).max()
    box = (-0.5, -0.3, -1.0, 0.5, 0.6, 1.0)
    clipped = BoxClip(box=box)(copy_of(scan))
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.8))
    draw_top_view(axes[0], scan, "원본 (정규화)", show_axes=True)
    draw_top_view(axes[1], clipped, f"BoxClip(box={box})", removed=removed_points(scan, clipped), show_axes=True)
    for ax in axes:
        ax.add_patch(Rectangle((box[0], box[1]), box[3] - box[0], box[4] - box[1], fill=False, ec="#0b0b0b", lw=1, ls="--"))
    add_legend(fig, with_removed=True)
    save_figure(fig, __file__)


if __name__ == "__main__":
    visualize()
