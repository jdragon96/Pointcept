import random

from pointcept.datasets.transform import TRANSFORMS


@TRANSFORMS.register_module()
class RandomMirror(object):
    def __init__(self, axis=0, teeth_per_side=8, p=0.5):
        self.axis = axis
        self.teeth_per_side = teeth_per_side
        self.p = p

    def __call__(self, data_dict):
        if random.random() > self.p:
            return data_dict
        if "coord" in data_dict.keys():
            data_dict["coord"][:, self.axis] *= -1
        if "normal" in data_dict.keys():
            data_dict["normal"][:, self.axis] *= -1
        if "segment" in data_dict.keys():
            data_dict["segment"] = self.swap_sides(data_dict["segment"])
        return data_dict

    def swap_sides(self, segment):
        one_side = (segment >= 1) & (segment <= self.teeth_per_side)
        other_side = (segment > self.teeth_per_side) & (segment <= 2 * self.teeth_per_side)
        swapped = segment.copy()
        swapped[one_side] += self.teeth_per_side
        swapped[other_side] -= self.teeth_per_side
        return swapped


def visualize():
    from plot_utils import add_legend, copy_of, draw_top_view, load_example_scan, plt, save_figure

    scan = load_example_scan()
    flipped_only = copy_of(scan)
    flipped_only["coord"][:, 0] *= -1
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.6))
    draw_top_view(axes[0], scan, "원본")
    draw_top_view(axes[1], flipped_only, "좌표만 반전 (라벨 그대로)")
    draw_top_view(axes[2], RandomMirror(p=1.0)(copy_of(scan)), "RandomMirror (라벨도 교환)")
    add_legend(fig)
    save_figure(fig, __file__)


if __name__ == "__main__":
    visualize()
