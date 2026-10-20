import random

import numpy as np
from scipy.spatial.transform import Rotation

from pointcept.datasets.transform import TRANSFORMS


@TRANSFORMS.register_module()
class RandomSO3(object):
    def __init__(self, center=None, p=1.0):
        self.center = center
        self.p = p

    def __call__(self, data_dict):
        if random.random() > self.p:
            return data_dict
        rotation = Rotation.random().as_matrix()
        if "coord" in data_dict.keys():
            coord = data_dict["coord"]
            center = self.get_center(coord)
            rotated = (coord - center) @ rotation.T + center
            data_dict["coord"] = rotated.astype(coord.dtype)
        if "normal" in data_dict.keys():
            normal = data_dict["normal"]
            rotated = normal @ rotation.T
            data_dict["normal"] = rotated.astype(normal.dtype)
        return data_dict

    def get_center(self, coord):
        if self.center is None:
            return (coord.min(axis=0) + coord.max(axis=0)) / 2
        return np.asarray(self.center)


def visualize():
    from plot_utils import add_legend, copy_of, draw_3d_view, load_example_scan, plt, save_figure

    scan = load_example_scan()
    fig = plt.figure(figsize=(10, 3.6))
    draw_3d_view(fig.add_subplot(1, 3, 1, projection="3d"), scan, "원본")
    for i in (2, 3):
        rotated = RandomSO3()(copy_of(scan))
        draw_3d_view(fig.add_subplot(1, 3, i, projection="3d"), rotated, f"RandomSO3 샘플 {i - 1}")
    add_legend(fig)
    save_figure(fig, __file__)


if __name__ == "__main__":
    visualize()
