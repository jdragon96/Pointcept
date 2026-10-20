import random

import numpy as np

from pointcept.datasets.transform import TRANSFORMS, index_operator


@TRANSFORMS.register_module()
class RandomSphereDrop(object):
    def __init__(self, num_spheres=(1, 3), radius_ratio=(0.02, 0.05), p=0.5):
        self.num_spheres = num_spheres
        self.radius_ratio = radius_ratio
        self.p = p

    def __call__(self, data_dict):
        if random.random() > self.p:
            return data_dict
        coord = data_dict["coord"]
        bbox_diagonal = np.linalg.norm(coord.max(axis=0) - coord.min(axis=0))
        keep = np.ones(len(coord), dtype=bool)
        for _ in range(random.randint(*self.num_spheres)):
            center = coord[np.random.randint(len(coord))]
            radius = np.random.uniform(*self.radius_ratio) * bbox_diagonal
            keep &= np.linalg.norm(coord - center, axis=1) > radius
        return index_operator(data_dict, keep)


def visualize():
    from plot_utils import add_legend, copy_of, draw_top_view, load_example_scan, plt, removed_points, save_figure

    scan = load_example_scan()
    augmentation = RandomSphereDrop(num_spheres=(3, 3), p=1.0)
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.6))
    draw_top_view(axes[0], scan, "원본")
    for i in (1, 2):
        dropped = augmentation(copy_of(scan))
        draw_top_view(axes[i], dropped, f"RandomSphereDrop 샘플 {i}", removed=removed_points(scan, dropped))
    add_legend(fig, with_removed=True)
    save_figure(fig, __file__)


if __name__ == "__main__":
    visualize()
