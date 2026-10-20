import numpy as np

from pointcept.datasets.transform import TRANSFORMS, index_operator


@TRANSFORMS.register_module()
class FarthestPointSample(object):
    def __init__(self, num_points=16000):
        self.num_points = num_points

    def __call__(self, data_dict):
        coord = data_dict["coord"]
        if len(coord) <= self.num_points:
            return data_dict
        index = self.farthest_point_sample(coord, self.num_points)
        return index_operator(data_dict, index)

    @staticmethod
    def farthest_point_sample(coord, num_points):
        x, y, z = coord.T.astype(np.float32)
        index = np.zeros(num_points, dtype=np.int64)
        index[0] = np.random.randint(len(coord))
        distance_to_sampled = np.full(len(coord), np.inf, dtype=np.float32)
        for i in range(1, num_points):
            last_x, last_y, last_z = x[index[i - 1]], y[index[i - 1]], z[index[i - 1]]
            distance_to_last = (x - last_x) ** 2 + (y - last_y) ** 2 + (z - last_z) ** 2
            np.minimum(distance_to_sampled, distance_to_last, out=distance_to_sampled)
            index[i] = np.argmax(distance_to_sampled)
        return index


def visualize():
    from plot_utils import add_legend, copy_of, draw_top_view, load_example_scan, plt, save_figure, subsample

    scan = load_example_scan()
    num_points = 1500
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.8), sharex=True, sharey=True)
    draw_top_view(axes[0], subsample(scan, num_points), f"랜덤 {num_points}점", size=2)
    draw_top_view(axes[1], FarthestPointSample(num_points)(copy_of(scan)), f"FarthestPointSample {num_points}점", size=2)
    add_legend(fig)
    save_figure(fig, __file__)


if __name__ == "__main__":
    visualize()
