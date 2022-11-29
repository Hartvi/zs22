import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import math

npr = np.array
rn_jesus = np.random.default_rng()


def e2p_2d_to_3d(x: np.ndarray, alpha=1):
    return alpha * np.array([x[0], x[1], 1])


def p2e_3d_to_2d(x: np.ndarray):
    el3 = x[2]
    return x[:2] / el3 if el3 != 0 else x[:2] * float("inf")


def arr_3d_to_2d(arr: np.ndarray):
    return np.apply_along_axis(func1d=p2e_3d_to_2d, axis=0, arr=arr)


def arr_2d_to_3d(arr: np.ndarray):
    return np.apply_along_axis(func1d=e2p_2d_to_3d, axis=0, arr=arr)


def apply_h_2d(h: np.ndarray, x_2d: np.ndarray):
    assert x_2d.shape[0] == 2, "Points must be 2d column vectors"
    x_3d = np.apply_along_axis(func1d=e2p_2d_to_3d, axis=0, arr=x_2d)
    y_3d = h @ x_3d
    return np.apply_along_axis(func1d=p2e_3d_to_2d, axis=0, arr=y_3d)


def select_points(book1, book2, correspondences):
    return book1[correspondences[:, 0]].T, book2[correspondences[:, 1]].T


def stopping_function(support, number_of_elements, p, n):
    w = support / number_of_elements
    Nmax = math.log(1-p) / math.log(1 - w**n)
    return Nmax


def find_homography(points_source, points_target):
    A = construct_A(points_source.T, points_target.T)
    u, s, vh = np.linalg.svd(A, full_matrices=True)

    # Solution to H is the last column of V, or last row of V transpose
    homography = vh[-1].reshape((3, 3))
    return homography / homography[2, 2]


def construct_A(points_source, points_target):
    assert points_source.shape == points_target.shape, "Shape does not match"
    num_points = points_source.shape[0]

    matrices = []
    for i in range(num_points):
        partial_A = construct_A_partial(points_source[i], points_target[i])
        matrices.append(partial_A)
    return np.concatenate(matrices, axis=0)


def construct_A_partial(point_source, point_target):
    x, y, z = point_source[0], point_source[1], 1
    x_t, y_t, z_t = point_target[0], point_target[1], 1

    A_partial = np.array([
        [0, 0, 0, -z_t * x, -z_t * y, -z_t * z, y_t * x, y_t * y, y_t * z],
        [z_t * x, z_t * y, z_t * z, 0, 0, 0, -x_t * x, -x_t * y, -x_t * z]
    ])
    return A_partial


def point_dist(p1, p2):
    return np.linalg.norm(p1-p2)


def inlier_criterium(arr: np.ndarray, theta):
    return (arr < theta).astype(int)


def find_inliers(original_points, H, target_points_2d, theta):
    pass


def get_errors(target_points, mapped_points):
    return np.apply_along_axis(func1d=np.linalg.norm, axis=0, arr=target_points - mapped_points)


def vanilla_ransac(book1: np.ndarray, book2: np.ndarray, theta=10, max_iters=1000, mode=0, Ha=None, get_a=False):
    assert book1.shape[0] == 2 and book2.shape[0] == 2, "Book points must be 2d column vectors"

    number_of_points = book1.shape[1]
    best_support = -float('inf')
    best_ha = np.eye(3)
    best_hb = np.eye(3)
    k = 0
    support = -float('inf')
    b1_3d = np.apply_along_axis(func1d=e2p_2d_to_3d, axis=0, arr=book1)
    if mode == 2:
        assert Ha is not None, "mode 2 has to have an Ha provided"
        b2_estimate_2d = apply_h_2d(Ha, book1)
        b12_errors = get_errors(book2, b2_estimate_2d)
        supporters = inlier_criterium(b12_errors, theta).astype(bool)
        Ha_book1_outliers = book1[:, ~supporters]
        Ha_book2_outliers = book2[:, ~supporters]
        Ha_book1_outliers_3d = b1_3d[:, ~supporters]
        support_a = np.sum(supporters, dtype=int)
        support = support_a
        pass
    while k < max_iters:
        if mode != 2:
            random_ass_choice = rn_jesus.choice(number_of_points, 4, replace=False)
            b1_choice = book1[:, random_ass_choice]
            b2_choice = book2[:, random_ass_choice]
            Ha = find_homography(points_source=b1_choice, points_target=b2_choice)
            b2_estimate_2d = apply_h_2d(Ha, book1)
            b12_errors = get_errors(book2, b2_estimate_2d)
            supporters = inlier_criterium(b12_errors, theta).astype(bool)
            Ha_book1_outliers = book1[:, ~supporters]
            Ha_book2_outliers = book2[:, ~supporters]
            # Ha_book1_outliers_3d = b1_3d[:, ~supporters]
            support_a = np.sum(supporters, dtype=int)
            support = support_a
        if mode == 1 or mode == 2:
            assert Ha_book1_outliers.shape[1] != 2, 'points must be in a row, not column'
            random_ass_choice = rn_jesus.choice(Ha_book1_outliers.shape[1], 3, replace=False)
            b1_choice = Ha_book1_outliers[:, random_ass_choice]
            b2_choice = Ha_book2_outliers[:, random_ass_choice]
            b1_choice_3d = arr_2d_to_3d(b1_choice)
            b2_choice_3d = arr_2d_to_3d(b2_choice)
            invHa = np.linalg.inv(Ha)
            b21_choice = invHa @ arr_2d_to_3d(b2_choice)
            b21_target = apply_h_2d(invHa, Ha_book2_outliers)
            u_s = [b1_choice_3d[:, _] for _ in range(b1_choice_3d.shape[1])]
            u_primes = [b21_choice[:, _] for _ in range(b21_choice.shape[1])]
            u1, u2, u3 = u_s
            u1p, u2p, u3p = u_primes
            v = np.cross(np.cross(u1, u1p), np.cross(u2, u2p))
            A = npr([(u1p[0]*v[2] - u1p[2]*v[0])*u1,
                     (u2p[0]*v[2] - u2p[2]*v[0])*u2,
                     (u3p[0]*v[2] - u3p[2]*v[0])*u3])
            b = npr([u1[0]*u1p[2] - u1[2]*u1p[0],
                     u2[0]*u2p[2] - u2[2]*u2p[0],
                     u3[0]*u3p[2] - u3[2]*u3p[0]])
            res = np.linalg.lstsq(A, b)
            a = res[0]
            H = np.eye(3) + v.reshape((-1,1)) @ np.expand_dims(a, axis=0)
            # Hb = Ha @ H
            b2_estimate_2d = apply_h_2d(H, Ha_book1_outliers)
            b12_errors = get_errors(b21_target, b2_estimate_2d)
            supporters = inlier_criterium(b12_errors, theta)
            support = np.sum(supporters) + support_a
        if support > best_support:
            best_support = support
            best_ha = Ha
            if mode != 0:
                best_hb = Ha @ H
                best_a = a
            print("support_a:", support_a, " support_b:", support-support_a)
            # print("h:", best_h, "  support:", support)

        k += 1
    if mode == 0:
        return best_ha
    elif mode == 1:
        return best_ha, best_hb
    else:
        if get_a:
            return best_hb, best_a
        else:
            return best_hb



# plt.scatter([111, 222] ,[111, 222])
book1 = np.loadtxt('books_u1.txt')
book2 = np.loadtxt('books_u2.txt')
book_m12 = np.loadtxt('books_m12.txt', dtype=int)
book1_sel, book2_sel = select_points(book1, book2, book_m12)
# print(book_m12.shape)
print('initial shape:', book1_sel.shape)
theta = 3
mode = 2
max_iters = 1000
if mode == 0:
    ha = vanilla_ransac(book1_sel, book2_sel, theta=theta, max_iters=max_iters, mode=mode)
    b2_estimation = apply_h_2d(ha, book1_sel)
    errs_a = get_errors(b2_estimation, book2_sel)
    inlier_indices_a = inlier_criterium(errs_a, theta=theta).astype(bool)
    book1_outliers_a = book1_sel[:, ~inlier_indices_a]
    book2_outliers_a = book2_sel[:, ~inlier_indices_a]
    print("original shape:", book1_sel.shape)
    print("outliers shape:", book1_outliers_a.shape)

    plt.scatter(book1_sel[0, :], book1_sel[1, :], s=10, color='red')
    plt.scatter(book1_outliers_a[0, :], book1_outliers_a[1, :], s=1, color=(0, 0, 0))
    # plt.scatter(book2_sel[0, :], book2_sel[1, :], s=1, color=(1,0,0))
    # plt.scatter(b2_estimation[0, :], b2_estimation[1, :], s=1, color=(0,1,0))
elif mode == 1:
    ha, hb = vanilla_ransac(book1_sel, book2_sel, theta=theta, max_iters=max_iters, mode=mode)
    b2_estimation_a = apply_h_2d(ha, book1_sel)
    errs_a = get_errors(b2_estimation_a, book2_sel)
    inlier_indices_a = inlier_criterium(errs_a, theta=theta).astype(bool)
    book1_outliers_a = book1_sel[:, ~inlier_indices_a]
    book2_outliers_a = book2_sel[:, ~inlier_indices_a]
    b2_estimation_b = apply_h_2d(hb, book1_outliers_a)
    # print("original shape:", book1_sel.shape)
    # print("outliers shape:", book1_outliers.shape)

    # plt.scatter(book1_sel[0, :], book1_sel[1, :], s=10, color='red')
    # plt.scatter(book1_outliers[0, :], book1_outliers[1, :], s=1, color=(0,0,0))
    plt.scatter(book2_sel[0, :], book2_sel[1, :], s=1, color=(1,0,0))
    plt.scatter(b2_estimation_a[0, :], b2_estimation_a[1, :], s=1, color=(0,1,0))
    plt.scatter(b2_estimation_b[0, :], b2_estimation_b[1, :], s=1, color=(0,0,1))
elif mode == 2:

    ha = vanilla_ransac(book1_sel, book2_sel, theta=theta, max_iters=max_iters, mode=0)
    b2_estimation_a = apply_h_2d(ha, book1_sel)
    errs_a = get_errors(b2_estimation_a, book2_sel)
    hb, a = vanilla_ransac(book1_sel, book2_sel, theta=theta, max_iters=max_iters, mode=mode, Ha=ha, get_a=True)
    b2_estimation_b = apply_h_2d(hb, book1_sel)
    errs_b = get_errors(b2_estimation_b, book2_sel)
    # print("book1_sel.shape", book1_sel.shape)
    # print(errs_a.tolist())
    # print(errs_b.tolist())
    # print((errs_a < errs_b).tolist())
    # print("errs_a.shape", errs_a.shape)
    # print("((errs_a < errs_b) & (errs_a < theta)).shape", ((errs_a < errs_b) & (errs_a < theta)).shape)
    # print("(errs_a < errs_b) sum", np.sum((errs_a < errs_b)))
    # print("((errs_a < errs_b) & (errs_a < theta)) sum", np.sum(((errs_a < errs_b) & (errs_a < theta))))
    # inliers1_a = book1_sel[:, (errs_a < errs_b) & (errs_a < theta)]
    inliers1_a = book1_sel[:, (errs_a < theta)]
    # print("inliers1_a.shape",inliers1_a.shape)
    inliers1_b = book1_sel[:, (errs_b < errs_a) & (errs_b < theta)]
    # inliers1_b = book1_sel[:, ~(errs_a < theta) & (errs_b < theta)]
    outliers1_all = book1_sel[:, (errs_b >= theta) & (errs_a >= theta)]
    inliers2_a = book2_sel[:, (errs_a < errs_b) & (errs_a < theta)]
    # inliers2_a = book2_sel[:, (errs_a < theta)]
    inliers2_b = book2_sel[:, (errs_b < errs_a) & (errs_b < theta)]
    # inliers2_b = book2_sel[:, ~(errs_a < theta) & (errs_b < theta)]
    outliers2_all = book2_sel[:, (errs_b >= theta) & (errs_a >= theta)]

    fig, axs = plt.subplots(2)
    ax1, ax2 = axs
    im1 = Image.open('im1.png')  # type: Image
    ax1.imshow(im1)
    for i in range(outliers1_all.shape[1]):
        ax1.plot([outliers1_all[0, i], outliers2_all[0, i]], [outliers1_all[1, i], outliers2_all[1, i]], color=(0, 0, 0))
    for i in range(min(inliers1_a.shape[1], inliers2_a.shape[1])):
        ax1.plot([inliers1_a[0, i], inliers2_a[0, i]], [inliers1_a[1, i], inliers2_a[1, i]], color=(1,0,0))
    for i in range(min(inliers1_b.shape[1], inliers2_b.shape[1])):
        ax1.plot([inliers1_b[0, i], inliers2_b[0, i]], [inliers1_b[1, i], inliers2_b[1, i]], color=(0, 1, 0))

    k = a[0] / a[1]
    b = a[2] / a[1]
    p1 = npr([-b/k, 0])
    # x = y/k - b/k
    p2 = npr([im1.size[1] / k - b / k, 648])
    ax1.plot([p1[0], p2[0]], [p1[1], p2[1]], color='magenta')

    im2 = Image.open('im2.png')  # type: Image
    ax2.imshow(im2)
    for i in range(outliers1_all.shape[1]):
        ax2.plot([outliers1_all[0, i], outliers2_all[0, i]], [outliers1_all[1, i], outliers2_all[1, i]], color=(0, 0, 0))
    for i in range(min(inliers1_a.shape[1], inliers2_a.shape[1])):
        ax2.plot([inliers1_a[0, i], inliers2_a[0, i]], [inliers1_a[1, i], inliers2_a[1, i]], color=(1,0,0))
    for i in range(min(inliers1_b.shape[1], inliers2_b.shape[1])):
        ax2.plot([inliers1_b[0, i], inliers2_b[0, i]], [inliers1_b[1, i], inliers2_b[1, i]], color=(0, 1, 0))

    # k = a[0] / a[1]
    # b = a[2] / a[1]
    p1 = apply_h_2d(ha, p1)
    # x = y/k - b/k
    p2 = apply_h_2d(ha, p2)
    ax2.plot([p1[0], p2[0]], [p1[1], p2[1]], color='magenta')
    # print(im.size[1])
    # inlier_indices_a = inlier_criterium(errs_a, theta=theta).astype(bool)
    # book1_outliers_a = book1_sel[:, ~inlier_indices_a]
    # book2_outliers_a = book2_sel[:, ~inlier_indices_a]
    # b2_estimation_inliers_a = b2_estimation_a[:, inlier_indices_a]
    # book1_inliers_a = book1_sel[:, inlier_indices_a]
    # book2_inliers_a = book2_sel[:, inlier_indices_a]
    #
    # # print(book1_outliers.shape)
    # # print(hb)
    # b2_estimation_b = apply_h_2d(hb, book1_outliers_a)
    # errs_b = get_errors(b2_estimation_b, book2_outliers_a)
    # inlier_indices_b = inlier_criterium(errs_b, theta=theta).astype(bool)
    # book1_outliers_b = book1_outliers_a[:, ~inlier_indices_b]
    # book2_outliers_b = book2_outliers_a[:, ~inlier_indices_b]
    # b2_estimation_inliers = b2_estimation_b[:, inlier_indices_b]
    # book1_inliers_b = book1_outliers_a[:, inlier_indices_b]
    # book2_inliers_b = book1_outliers_a[:, inlier_indices_b]

    # print("original shape:", book1_sel.shape)
    # print("outliers shape:", book1_outliers.shape)

    # plt.scatter(book1_sel[0, :], book1_sel[1, :], s=10, color='red')
    # plt.scatter(book1_outliers[0, :], book1_outliers[1, :], s=1, color=(0,0,0))
    visualize_points_only = False
    # if visualize_points_only:
    #     plt.scatter(book2_outliers_a[0, :], book2_outliers_a[1, :], s=10, color=(0, 0, 0))
    #     plt.scatter(book2_sel[0, :], book2_sel[1, :], s=1, color=(1,0,0))
    #     plt.scatter(b2_estimation_inliers_a[0, :], b2_estimation_inliers_a[1, :], s=1, color=(0, 1, 0))
    #     plt.scatter(b2_estimation_b[0, :], b2_estimation_b[1, :], s=1, color=(0,0,1))
    #     plt.legend(["Ha outliers", 'reference', 'Ha estimate', "Hb estimate"])
    # else:
        # pass



plt.show()

