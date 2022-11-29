import numpy as np
from matplotlib import pyplot as plt

npr = np.array
rn_jesus = np.random.default_rng()


def e2p_2d_to_3d(x: np.ndarray, alpha=1):
    return alpha * np.array([x[0], x[1], 1])


def p2e_3d_to_2d(x: np.ndarray):
    el3 = x[2]
    return x[:2] / el3 if el3 != 0 else x[:2] * float("inf")


def normalized_point_error(normalized_x: np.ndarray, line: np.ndarray):
    assert normalized_x[2] == 1
    assert np.dot(line[:2], line[:2]) == 1
    return np.dot(normalized_x, line)


# def lsq(x, y):
#     A = np.concatenate((np.ones((x.shape[0], 1)), x.reshape(-1, 1)), axis=1)
#     t = np.linalg.inv(A.T @ A) @ A.T @ -y.reshape(-1, 1)
#     return t[0], t[1]


def ran_support(theta=3):
    def parametrized_loss(eps):
        # print("eps:", eps)
        return 1.0 if abs(eps) <= theta else 0

    return parametrized_loss


def mle_support(theta=3):
    inv_theta_sqr = 1.0 / (theta * theta)
    # print(inv_theta_sqr)

    def parametrized_loss(eps):
        # print("eps:", eps)
        ret = max(0, 1.0 - eps * eps * inv_theta_sqr)
        # ret = 1.0 if abs(eps) <= theta else 0
        # if ret < 0:
        # print("mle error: ", ret)
        return ret

    return parametrized_loss


def get_error(points_3d, l, support_func, verbose=False):
    den = (l[0] ** 2 + l[1] ** 2) ** (1 / 2)
    errs = l @ points_3d  # type: np.ndarray
    if verbose:
        print("support:", errs)
    errs = np.abs(errs / den)
    # print(min(errs))

    # print(vec_err_func(errs))
    individual_support = support_func(errs)
    if verbose:
        print("individual_support:", individual_support)
    support = np.sum(individual_support)
    return support


def fit_line_eigen(sample_2d):
    u = np.apply_along_axis(func1d=np.mean, axis=1, arr=sample_2d)
    centered_points_2d = np.apply_along_axis(func1d=lambda x: x - u, axis=0, arr=sample_2d)
    eigen_yolks, eig_veggies = np.linalg.eig(
        centered_points_2d @ centered_points_2d.T)  # smallest eigen vector last confirmed
    n = eig_veggies[0]
    d = -np.dot(n, u)
    l = npr([*n, d])
    return l


def fit_line_lsq(sample_2d):
    b = -sample_2d[1, :]
    a = np.ones(sample_2d.shape).T
    a[:, 0] = sample_2d[0, :]
    lsq_res = np.linalg.lstsq(a, b)
    t = lsq_res[0]
    l = npr([t[0], 1, t[1]])
    return l


def ransack_the_village(points_2d: np.ndarray, theta=50, max_iters=1000, sample_size=100, mode='ransac'):
    support_func = None
    if mode.lower() == 'ransac':
        support_func = ran_support(theta)
    elif mode.lower() == 'mlesac':
        support_func = mle_support(theta)
    vec_support_func = np.vectorize(support_func, otypes=[float])
    points_3d = np.apply_along_axis(func1d=e2p_2d_to_3d, axis=0, arr=points_2d)
    # print(points_3d)
    # print(points_2d)
    number_of_points = points_2d.shape[1]
    best_support = -float('inf')
    best_line = npr([0, 0, 0])
    k = 0
    while k < max_iters:
        random_ass_choice = rn_jesus.choice(number_of_points, sample_size, replace=False)
        random_ass_sample_2d = points_2d[:, random_ass_choice]
        # random_ass_sample_3d = points_3d[:, random_ass_choice]



        l = fit_line_eigen(random_ass_sample_2d)

        support = get_error(points_3d, l, vec_support_func)

        if support > best_support:
            best_support = support
            best_line = l
            # print("mode:", mode, "line:", l, "  support:", support)

        k += 1
    return best_line


def get_inliers(l: np.ndarray, points_3d: np.ndarray, theta):
    support_func = ran_support(theta)
    vec_support_func = np.vectorize(support_func, otypes=[bool])
    den = (l[0] ** 2 + l[1] ** 2) ** (1 / 2)
    errs = l @ points_3d  # type: np.ndarray
    errs = np.abs(errs / den)
    # print(min(errs))

    # print(vec_err_func(errs))
    individual_support = vec_support_func(errs)
    # print(individual_support)
    # print(individual_support.shape)
    # exit(1)
    # den = (l[0] ** 2 + l[1] ** 2) ** (1 / 2)
    # errs = l @ points_3d  # type: np.ndarray
    # errs = errs / den
    #
    # indices = vec_err_func(errs)
    return individual_support


def plot_algorithm(points_2d):
    # setup:
    points_3d = np.apply_along_axis(func1d=e2p_2d_to_3d, axis=0, arr=points_2d)
    edge_bounds = [[min(points_2d[0, :]), min(points_2d[1, :])],
                   [max(points_2d[0, :]), max(points_2d[1, :])]]
    edge_points = [edge_bounds[0], [edge_bounds[1][0], edge_bounds[0][1]], edge_bounds[1],
                   [edge_bounds[0][0], edge_bounds[1][1]], edge_bounds[0]]
    edge_points_3d = [e2p_2d_to_3d(_) for _ in edge_points]
    edge_lines = [np.cross(edge_points_3d[i - 1], edge_points_3d[i]) for i in range(1, len(edge_points_3d))]
    n = 3
    max_iters = 1000
    theta = 3

    # line ORIGINAL
    orig = npr([-10, 3, 1200])
    orig_edges = [np.cross(orig, edge_lines[_]) for _ in range(len(edge_lines))]
    orig_edges_2d = npr(list(filter(lambda x: 0 <= x[0] <= edge_bounds[1][0] + 1 and
                                                0 <= x[1] <= edge_bounds[1][1] + 1,
                                      [p2e_3d_to_2d(_) for _ in orig_edges])))
    plt.plot(orig_edges_2d[:, 0], orig_edges_2d[:, 1], color='black', label='ORIGINAL LINE')


    # NAIVE LSQ
    inliers = points_2d
    lsq_fit = fit_line_lsq(inliers)
    lsq_edges = [np.cross(lsq_fit, edge_lines[_]) for _ in range(len(edge_lines))]
    lsq_edges_2d = npr(list(filter(lambda x: 0 <= x[0] <= edge_bounds[1][0] + 1 and
                                             0 <= x[1] <= edge_bounds[1][1] + 1,
                                   [p2e_3d_to_2d(_) for _ in lsq_edges])))
    plt.plot(lsq_edges_2d[:, 0], lsq_edges_2d[:, 1], color='purple', label='LSQ')

    # RANSAC
    ransac_fit = ransack_the_village(points_2d, theta=theta, max_iters=max_iters, sample_size=n, mode='ransac')
    ransac_edges = [np.cross(ransac_fit, edge_lines[_]) for _ in range(len(edge_lines))]
    ransac_edges_2d = npr(list(filter(lambda x: 0 <= x[0] <= edge_bounds[1][0] + 1 and
                                                0 <= x[1] <= edge_bounds[1][1] + 1,
                                      [p2e_3d_to_2d(_) for _ in ransac_edges])))
    # ransac_edges_2d = npr([p2e_3d_to_2d(_) for _ in ransac_edges])
    plt.plot(ransac_edges_2d[:, 0], ransac_edges_2d[:, 1], color='red', label='RANSAC')

    # RANSAC + LSQ
    inlier_indices = get_inliers(ransac_fit, points_3d, theta)
    inliers = points_2d[:, inlier_indices]
    # plt.scatter(inliers[0, :], inliers[1,:], s=50, color='cyan')
    # print(inliers.shape)
    RSQ_fit = fit_line_lsq(inliers)
    RSQ_edges = [np.cross(RSQ_fit, edge_lines[_]) for _ in range(len(edge_lines))]
    RSQ_edges_2d = npr(list(filter(lambda x: 0 <= x[0] <= edge_bounds[1][0] + 1 and
                                                0 <= x[1] <= edge_bounds[1][1] + 1,
                                      [p2e_3d_to_2d(_) for _ in RSQ_edges])))
    plt.plot(RSQ_edges_2d[:, 0], RSQ_edges_2d[:, 1], color='green', label='RANSAC + LSQ')
    # print('inliers:', inliers)


    # l = npr([-0.95087892,   0.30956305, 117.85452607])
    # vec_support_func = np.vectorize(mle_support(theta), otypes=[float])
    # support = get_error(points_3d, l, vec_support_func, verbose=True)
    # print('ransac line support: ', support)

    # MLESAC
    mlesac_fit = ransack_the_village(points_2d, theta=theta, max_iters=max_iters, sample_size=n, mode='mlesac')
    mlesac_edges = [np.cross(mlesac_fit, edge_lines[_]) for _ in range(len(edge_lines))]
    mlesac_edges_2d = npr(list(filter(lambda x: 0 <= x[0] <= edge_bounds[1][0] + 1 and
                                                0 <= x[1] <= edge_bounds[1][1] + 1,
                                      [p2e_3d_to_2d(_) for _ in mlesac_edges])))
    # mlesac_edges_2d = npr([p2e_3d_to_2d(_) for _ in mlesac_edges])
    plt.plot(mlesac_edges_2d[:, 0], mlesac_edges_2d[:, 1], color='blue', label='MLESAC')

    # MLESAC + LSQ
    inlier_indices = get_inliers(mlesac_fit, points_3d, theta)
    inliers = points_2d[:, inlier_indices]
    MSQ_fit = fit_line_lsq(inliers)
    MSQ_edges = [np.cross(MSQ_fit, edge_lines[_]) for _ in range(len(edge_lines))]
    MSQ_edges_2d = npr(list(filter(lambda x: 0 <= x[0] <= edge_bounds[1][0] + 1 and
                                             0 <= x[1] <= edge_bounds[1][1] + 1,
                                   [p2e_3d_to_2d(_) for _ in MSQ_edges])))
    plt.plot(MSQ_edges_2d[:, 0], MSQ_edges_2d[:, 1], color='yellow', label='MLESAC + LSQ')


if __name__ == "__main__":
    points_2d = np.loadtxt('linefit_1.txt').T  # type: np.ndarray

    plot_algorithm(points_2d)
    # edge_bounds = [[min(points_2d[0, :]), min(points_2d[1, :])],
    #                [max(points_2d[0, :]), max(points_2d[1, :])]]
    # edge_points = [edge_bounds[0], [edge_bounds[1][0], edge_bounds[0][1]], edge_bounds[1],
    #                [edge_bounds[0][0], edge_bounds[1][1]], edge_bounds[0]]
    # edge_points_3d = [e2p_2d_to_3d(_) for _ in edge_points]
    # edge_lines = [np.cross(edge_points_3d[i - 1], edge_points_3d[i]) for i in range(1, len(edge_points_3d))]
    # best_line = ransack_the_village(points_2d, theta=5, max_iters=10, sample_size=5, mode='mlesac')
    # line_edges = [np.cross(best_line, edge_lines[_]) for _ in range(len(edge_lines))]
    # # print(line_edges)
    # line_edges_2d = npr(list(filter(lambda x: 0 <= x[0] <= edge_bounds[1][0] + 1 and
    #                                           0 <= x[1] <= edge_bounds[1][1] + 1,
    #                                 [p2e_3d_to_2d(_) for _ in line_edges])))
    # # line_edges_2d = npr([p2e_3d_to_2d(_) for _ in line_edges])
    # plt.plot(line_edges_2d[:, 0], line_edges_2d[:, 1], color='red')

    plt.scatter(points_2d[0, :], points_2d[1, :], s=5, color='black')
    plt.legend()
    plt.show()
