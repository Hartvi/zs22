import numpy as np
npr = np.array

from matplotlib import pyplot as plt


def e2p_2d_to_3d(x: np.ndarray, alpha=1):
    return alpha*np.array([x[0], x[1], 1])


def p2e_3d_to_2d(x: np.ndarray):
    el3 = x[2]
    return x[:2] / el3 if el3 != 0 else x[:2]*float("inf")


# [x1, y1, w1] = W @ uwave
#  => [x'/w', y'/w']


def get_y_column(arr):
    return arr[:, 1]


def get_x_column(arr):
    return arr[:, 0]


class PlotPipeline:
    def __init__(self, plot_preprocessor=None):
        self.funcs = list()  # list of funcs
        self.kwargs = list()  # type: list[dict]
        self.plot_preprocessor = plot_preprocessor

    def add(self, f, **kw):
        self.funcs.append(f)
        self.kwargs.append(kw)

    def plot(self):
        for f, kw in zip(self.funcs, self.kwargs):
            if self.plot_preprocessor is not None:
                kw['x'], kw['y'] = self.plot_preprocessor(kw)  # must contain x, y
            x, y = kw['x'], kw['y']
            del kw['x']
            del kw['y']
            # y = 600*np.ones(y.shape) - y
            f(x, y, **kw)  # plot
        plt.gca().invert_yaxis()
        plt.show()


def my_preprocessor(d: dict):
    xy = npr([d['x'], d['y']])
    if len(xy.shape) == 1:
        xy = npr([[xy[0]], [xy[1]]])
    xy_3d = npr([e2p_2d_to_3d(xy[:, i]) for i in range(xy.shape[1])])
    tmp = (H @ xy_3d.transpose()).transpose()
    return npr([p2e_3d_to_2d(_) for _ in tmp]).transpose()


H = npr([[1, 0.1, 0], [0.1, 1, 0], [0.004, 0.002, 1]])
use_mat = False
plot_pipeline = PlotPipeline(my_preprocessor if use_mat else None)

img_bounds = npr([[1, 1], [800, 600]])
real_bounds_2d = np.array([img_bounds[0], [img_bounds[0][0], img_bounds[1][1]], img_bounds[1], [img_bounds[1][0], img_bounds[0][1]], img_bounds[0]])


## 2 pairs of points
pair1 = [[100, 100], [100, 150]]
# pair1 = [[100, 100], [200, 150]]
pair2 = [[500, 100], [300, 400]]
pairs_2d = np.array([pair1, pair2])
pairs_hom = npr([[e2p_2d_to_3d(p) for p in pr] for pr in pairs_2d])
lines_hom = [np.cross(hom_pr[0], hom_pr[1]) for hom_pr in pairs_hom]
# print(lines)
intersection_hom = np.cross(lines_hom[0], lines_hom[1])
intersection_2d = p2e_3d_to_2d(intersection_hom)

plot_pipeline.add(plt.plot, x=get_x_column(real_bounds_2d), y=get_y_column(real_bounds_2d), color="purple")
# plt.plot(get_x_column(real_bounds_2d), get_y_column(real_bounds_2d), color="purple")
if (img_bounds[0] <= intersection_2d).all() and (intersection_2d <= img_bounds[1]).all():
    plot_pipeline.add(plt.scatter, x=intersection_2d[0], y=intersection_2d[1])
    # plt.scatter(intersection_2d[0], intersection_2d[1])


cols = "rg"
lines = pairs_2d
for k,l in enumerate(lines):
    plot_pipeline.add(plt.scatter, x=get_x_column(l), y=get_y_column(l), color=cols[k])
    # plt.scatter(get_x_column(l), get_y_column(l), color=cols[k])

real_bounds_3d = list(map(lambda x: e2p_2d_to_3d(x), real_bounds_2d))
boundary_lines = [np.cross(real_bounds_3d[i], real_bounds_3d[i+1]) for i in range(len(real_bounds_2d) - 1)]
edge_points = list()
for bl in boundary_lines:
    for l in lines_hom:
        edge_points.append(np.cross(bl, l))

# limit x to the interval [1, 800] and y to [1, 600]
edge_points_2d = np.array(list(filter(lambda x: 1 <= x[0] <= 800 and 1 <= x[1] <= 600, map(lambda x: p2e_3d_to_2d(x), edge_points))))
real_bounds_hom = np.array(list(map(lambda x: e2p_2d_to_3d(x), edge_points_2d)))

render_line_indices = list()
edge_len = len(real_bounds_hom)
for i in range(edge_len):
    for k in range(i):
        res = np.dot(np.cross(real_bounds_hom[i], real_bounds_hom[k]), intersection_hom)
        if abs(res) < 1e-3:
            indices = np.array([i, k])
            line_ends = edge_points_2d[indices]
            xs = get_x_column(line_ends)
            ys = get_y_column(line_ends)
            plot_pipeline.add(plt.plot, x=xs, y=ys, color="orange", zorder=0) # plt.plot(xs, ys, color="orange", zorder=0)


plot_pipeline.plot()



