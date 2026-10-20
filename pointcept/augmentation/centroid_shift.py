import numpy as np
import torch

from pointcept.datasets.transform import TRANSFORMS


@TRANSFORMS.register_module()
class CentroidShift(object):
    def __init__(self, normalize=False):
        self.normalize = normalize

    def __call__(self, data_dict):
        if "coord" in data_dict.keys():
            coord = data_dict["coord"]
            if isinstance(coord, torch.Tensor):
                coord = coord - coord.mean(dim=0, keepdim=True)
                if self.normalize:
                    coord = coord / coord.norm(dim=1).max()
            else:
                coord = coord - np.mean(coord, axis=0, keepdims=True)
                if self.normalize:
                    coord = coord / np.linalg.norm(coord, axis=1).max()
            data_dict["coord"] = coord
        return data_dict


def visualize():
    from plot_utils import add_legend, copy_of, draw_top_view, load_example_scan, plt, save_figure

    scan = load_example_scan()
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.6))
    draw_top_view(axes[0], scan, "원본 (mm)", show_axes=True)
    draw_top_view(axes[1], CentroidShift()(copy_of(scan)), "CentroidShift()", show_axes=True)
    draw_top_view(axes[2], CentroidShift(normalize=True)(copy_of(scan)), "CentroidShift(normalize=True)", show_axes=True)
    add_legend(fig)
    save_figure(fig, __file__)


if __name__ == "__main__":
    visualize()
