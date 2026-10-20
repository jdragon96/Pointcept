import random

import numpy as np

from pointcept.datasets.transform import TRANSFORMS, index_operator


@TRANSFORMS.register_module()
class RandomPlaneCrop(object):
    def __init__(self, max_drop_ratio=0.3, p=0.3):
        self.max_drop_ratio = max_drop_ratio
        self.p = p

    def __call__(self, data_dict):
        if random.random() > self.p:
            return data_dict
        coord = data_dict["coord"]
        direction = np.random.randn(3)
        direction /= np.linalg.norm(direction)
        height = coord @ direction
        drop_ratio = np.random.uniform(0, self.max_drop_ratio)
        keep = height <= np.quantile(height, 1 - drop_ratio)
        return index_operator(data_dict, keep)


def visualize():
    from plot_utils import add_legend, copy_of, draw_top_view, load_example_scan, plt, removed_points, save_figure

    scan = load_example_scan()
    augmentation = RandomPlaneCrop(p=1.0)
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.6))
    draw_top_view(axes[0], scan, "원본")
    for i in (1, 2):
        cropped = augmentation(copy_of(scan))
        draw_top_view(axes[i], cropped, f"RandomPlaneCrop 샘플 {i}", removed=removed_points(scan, cropped))
    add_legend(fig, with_removed=True)
    save_figure(fig, __file__)


if __name__ == "__main__":
    visualize()
