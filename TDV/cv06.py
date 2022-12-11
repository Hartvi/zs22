import PIL.Image
import numpy as np
import os
from scipy import optimize
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
from PIL import Image
import math
import sympy as sp
# from p5.python import p5
import p5
# print()
# import ADAM_tools as tools
import tools

npr = np.array

rng_seed = 12345  # 1234 were okayish
rng = np.random.default_rng(rng_seed)
K = np.loadtxt('/home/hartvi/zs22/TDV/scene_1/K.txt')
invK = np.linalg.inv(K)
invTK = np.linalg.inv(K.T)


def get_sampson_errs_func(R, u1i, u2i):
    # print("R, t:", R, t)
    def err_func(args):
        rot_vec = args[:3]
        new_t = args[3:6]
        # tmp_R = tools.rodrigues(theta, axis)
        tmp_R = Rotation.from_rotvec(rotvec=rot_vec).as_matrix()
        new_R = tmp_R @ R
        F = invK.T @ tools.sqc(-new_t) @ new_R @ invK
        errs = tools.err_F_sampson(F, u1i, u2i)
        return np.sum(errs)
    return err_func


def get_inliers(errs, threshold):
    return np.abs(errs) < threshold**2


def eval_mle(u1p, u2p, F, threshold):
    errs = tools.err_F_sampson(F, u1p, u2p)
    inlier_indices = get_inliers(errs, threshold)
    errs = 1 - errs/threshold**2
    errs[errs < 0] = 0
    u1i, u2i = u1p[:, inlier_indices], u2p[:, inlier_indices]
    return u1i, u2i, np.sum(errs)


def ransac(u1, u2, threshold=3):
    # u1 = invK @ u1
    # u2 = (u2.T @ invK.T).T
    P1 = np.eye(3, 4)
    k = 0
    num_of_points = u1.shape[1]
    max_iters = 50
    best_E = np.eye(3)
    best_F = np.eye(3)
    best_R = np.eye(3)
    best_t = np.zeros(3)
    best_P2 = np.eye(3, 4)
    best_support = -float('inf')
    while k < max_iters:
        rng_choice = rng.choice(num_of_points, 5)
        u1p5, u2p5 = u1[:, rng_choice], u2[:, rng_choice]
        u1p = invK @ u1p5
        u2p = (u2p5.T @ invTK).T
        Es = p5.p5gb(u1p, u2p)
        for E in Es:
            R, t = tools.EutoRt(E, u1p5, u2p5)
            if np.linalg.det(R) < 0:
                print("DET was -1")
                continue
            F = invK.T @ E @ invK
            u1i, u2i, support = eval_mle(u1, u2, F, threshold)
            P2 = np.hstack((R, t))
            X = tools.Pu2X(P1, P2, u1i, u2i)
            X2 = P2 @ X

            in_front_mask = np.logical_and(X[2] > 0, X2[2] > 0)
            u1i, u2i = u1i[:, in_front_mask], u2i[:, in_front_mask]
            ui1, u2i, support = eval_mle(u1i, u2i, F, threshold)

            if support > best_support:
                max_iters = tools.Nmax(0.99, u1i.shape[1] / u1.shape[1], 5)
                best_support = support
                best_F = F
                best_R = R
                best_t = t
                # max_iters = tools.Nmax(success_prob=0.8, inlier_ratio=float(u1i.shape[1])/ num_of_points, number_of_params=5)
                print("max iters:", max_iters, "best iter:", k, " support:", support, " inliers: ", u1i.shape[1])
        k += 1

    u1i, u2i, support = eval_mle(u1, u2, best_F, threshold)
    print("number of inliers after ransac", u1i.shape[1])

    ig = np.zeros(6)
    ig[3:6] = best_t.flatten()
    xopt = optimize.fmin(func=get_sampson_errs_func(best_R, u1i, u2i), x0=ig)
    rot_vec = xopt[:3]
    opt_t = xopt[3:6].reshape((-1, 1))
    opt_R = Rotation.from_rotvec(rotvec=rot_vec).as_matrix() @ best_R
    opt_F = tools.calc_F(invK=invK, t=opt_t.flatten(), R=opt_R)
    opt_P2 = np.hstack([opt_R, -opt_t])

    # get stuff for the next task
    errs = tools.err_F_sampson(opt_F, u1, u2)
    inlier_mask = get_inliers(errs, threshold)
    inlier_indices = np.where(inlier_mask)[0]
    print("inlier_indices.shape:", inlier_indices.shape)

    u1i = u1[:, inlier_indices]
    u2i = u2[:, inlier_indices]
    u1i, u2i = tools.u_correct_sampson(opt_F, u1i, u2i)
    # print(opt_P2)
    Xu = tools.Pu2X(K @ P1, K @ opt_P2, u1i, u2i)
    return opt_F, opt_P2, inlier_indices, Xu


def plot_lines(im1, im2):
    root_path = "/home/hartvi/zs22/TDV/scene_1"
    corresp = np.loadtxt(os.path.join(root_path, "corresp", f"m_{im1}_{im2}.txt"), dtype=int)
    points1 = np.loadtxt(os.path.join(root_path, "corresp", f"u_{im1}.txt"))
    points2 = np.loadtxt(os.path.join(root_path, "corresp", f"u_{im2}.txt"))
    img1 = os.path.join(root_path, "images", f"{im1}.jpg")
    img2 = os.path.join(root_path, "images", f"{im2}.jpg")
    u1 = tools.e2p(points1[corresp[:, 0], :].T)
    u2 = tools.e2p(points2[corresp[:, 1], :].T)
    # optimize on the final inliers
    F, P2, inlier_indices, Xu = ransac(u1, u2, threshold=3)
    print("final F: ", F)
    print("u1 shape:", u1.shape)
    errs = tools.err_F_sampson(F, u1, u2)

    # begin get inliers
    contributions = tools.err_contributions(errs)
    inlier_indices = contributions > 0
    print("number of inliers after optimization:", np.sum(inlier_indices))
    u1i, u2i = u1[:, inlier_indices], u2[:, inlier_indices]
    l1 = F.T @ u2i
    l2 = F @ u1i
    x = np.linspace(0, 2816, 2816)

    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(PIL.Image.open(img1))
    axs[1].imshow(PIL.Image.open(img2))

    # print(l1.shape)
    ys2 = np.clip(npr([-(l2[0, i] / l2[1, i]) * x - (l2[2, i] / l2[1, i]) for i in range(l2.shape[1])]), 0, 1880)
    ys1 = np.clip(npr([-(l1[0, i] / l1[1, i]) * x - (l1[2, i] / l1[1, i]) for i in range(l1.shape[1])]), 0, 1880)
    # print(ys1.shape)
    for i in range(0, l1.shape[1], 100):
        axs[0].plot(x, ys1[i], color=(1, 0, 0))
        axs[1].plot(x, ys2[i], color=(0, 0, 1))

    # for i in range(len(correspondences_good)):
    #     ax.plot([points_1[0, correspondences_good[i, 0]], points_2[0, correspondences_good[i, 1]]],
    #             [points_1[1, correspondences_good[i, 0]], points_2[1, correspondences_good[i, 1]]], c='r',
    #             linewidth=0.3)
    #     # ax.scatter(points_1[0,correspondences[i,0]],points_1[1,correspondences[i,0]],c='r',linewidth=0.2) #,points_2[:,correspondences[i,1]]
    #     print(i)
    plt.show()


if __name__ == "__main__":
    plot_lines("01", "02")
    pass
