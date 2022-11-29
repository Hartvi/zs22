from numpy.core.defchararray import replace
from numpy.core.numeric import indices
import utils as ut
import numpy as np
from numpy.linalg import det, lstsq, norm, inv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from p5 import p5

# between pics 5 and 6


VERBOSE = True

# load interest points
u1_orig = np.loadtxt('data/points/u_01.txt')  # points on pic1
u2_orig = np.loadtxt('data/points/u_02.txt')  # points on pic2

K = np.array([[2080, 0, 1421],
              [0, 2080, 957],
              [0, 0, 1]])

# K undone points (calibrated world for E matrix)
u1 = ut.e2p(u1_orig.T)
u1 = np.matmul(inv(K), u1)
u1 = ut.p2e(u1).T

u2 = ut.e2p(u2_orig.T)
u2 = np.matmul(inv(K), u2)
u2 = ut.p2e(u2).T

# load tentative correspondencies
m12 = np.loadtxt('data/corrs/m_01_02.txt', dtype='int')

# create a list of correspondencies
corr12 = []
for pair in m12:
    corr = (np.reshape(u1_orig[pair[0]], [2, 1]), np.reshape(u2_orig[pair[1]], [2, 1]))
    corr12.append(corr)

rng = np.random.default_rng()

top_n_inliers = 0
cnt = 0
# ---- RANSAC ----


for i in range(100):
    five_indices = rng.choice(m12, 5, replace=False)
    u1p = np.zeros([2, 5])
    u2p = np.zeros([2, 5])

    # construct 5 and 5 points corresponding points for p5gb
    for n, idx in enumerate(five_indices):
        u1p[:, n] = u1[idx[0]]
        u2p[:, n] = u2[idx[1]]

    u1p = np.vstack([u1p, np.ones([1, 5])])
    u2p = np.vstack([u2p, np.ones([1, 5])])
    np.set_printoptions(suppress=True)

    # get maybe E matrices from external p5 algorithm
    maybe_Es = p5.p5gb(u1p, u2p)

    if VERBOSE:
        print("The number of possible Es: ", len(maybe_Es))
    for maybe_E in maybe_Es: np.set_printoptions(suppress=True)
    continue

if det(U) < 0:
    U = -U
if det(V_trans.T) < 0:
    V_trans = - V_trans

W = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
Ra = U@ W@ V_trans)
Rb = U@ W.T@ V_trans)
ta = U[:, -1].reshape(-1, 1)  # third column of U

P1 = np.hstack([np.eye(3), np.zeros([3, 1])])
# four possible solutions for the second camera
P2s = [np.hstack([Ra, ta]), np.hstack([Ra, -ta]), np.hstack([Rb, ta]), np.hstack([Rb, -ta])]

# now check cheirality constraint with a single point

x1 = u1p[:, 0].reshape(-1, 1)
x2 = u2p[:, 0].reshape(-1, 1)
P2 = None

for P2_ in P2s:
    valid_P = True
    for idx in range(np.shape(u1p)[1]):
        row_1 = x1[0] * P1[2, :] - P1[0, :]
        row_2 = x1[1] * P1[2, :] - P1[1, :]
        row_3 = x2[0] * P2_[2, :] - P2_[0, :]
        row_4 = x2[1] * P2_[2, :] - P2_[1, :]

        D = np.vstack([row_1, row_2, row_3, row_4])
        Q = D.T @ D
        U_q, D_q, V_q_trans = np.linalg.svd(Q)
        X = U_q[:, -1]  # the last column of U is triangulated real-life point
        X = ut.e2p(ut.p2e(X))

        X_P1 = P1 @ X
        X_P2 = P2_ @ X

        if X_P1[2] < 0 or X_P2[2] < 0:
            valid_P = False
            break

    if valid_P is True:
        break

if valid_P is False:
    if VERBOSE:
        print("BADBADBADBADBABDAD ")
    break

maybe_F = inv(K).T @ maybe_E @ inv(K)
threshold = 5  # 3 pixels threshold
# for #each u1_i and u2_i from corr pair
maybe_inliers = []
for corr in corr12:
    u1_i = ut.e2p(corr[0])
    u2_i = ut.e2p(corr[1])

    # print("corr: \n", corr)
    # print("u1_i: \n", u1_i)
    # print("u2_i: \n", u2_i)

    # epipolar line on the left pic
    l1 = np.matmul(maybe_F.T, u2_i)

    # epipolar line on the right pic
    l2 = np.matmul(maybe_F, u1_i)

    # distance of u2 from epipolar line l2
    e_l = abs((l2[0] * u2_i[0] + l2[1] * u2_i[1] + l2[2])) / (np.sqrt(l2[0] ** 2 + l2[1] ** 2))

    # distance of u1 from eipolar line l1
    e_r = abs((l1[0] * u1_i[0] + l1[1] * u1_i[1] + l1[2])) / (np.sqrt(l1[0] ** 2 + l1[1] ** 2))

    avg_e = (e_l + e_r) / 2

    if avg_e <= threshold:
        cnt += 1  # TODO probably not needed ?
        maybe_inliers.append(corr)

if len(maybe_inliers) >= top_n_inliers:
    top_n_inliers = len(maybe_inliers)
    inliers = maybe_inliers
    F = maybe_F

print("top_n_inliers: \n", top_n_inliers)

# just some plotting stuff
img1 = mpimg.imread("imgs/01.jpg")
img2 = mpimg.imread("imgs/02.jpg")
fig, axs = plt.subplots(1, 2)
axs[0].imshow(np.dot(img1[..., :3], [0.33, 0.33, 0.33]), cmap='gray')
axs[1].imshow(np.dot(img2[..., :3], [0.33, 0.33, 0.33]), cmap='gray')

# the line plotting is retarded, because of the
# inverting of the axis and default matplotlib plotting style

# y = (best_a[0]/best_a[1]) * x + (best_a[2]/best_a[1])
# plt.plot(x,y, color="m", lw=2)
x = np.linspace(0, 2816, 2816)
clrs = ["#de482a", "#de7e2a", "#deae2a", "#69de2a", "#2ade9c", "#2a8ade", "#4f29d9", "#ab24d4", "#d424cb", "#871258", ]
cnt = 0
for i, inlier in enumerate(inliers):
    axs[0].plot(inlier[0][0], inlier[0][1], 'o', markersize=2, color="tab:green")
    axs[0].plot([inlier[0][0], inlier[1][0]], [inlier[0][1], inlier[1][1]], color="tab:green", lw=1)

    if i % 200 == 0 and cnt < len(clrs):
        l1 = F.T @ ut.e2p(inlier[1])
        y = -(l1[0] / l1[1]) * x - (l1[2] / l1[1])
        axs[0].plot(x, y, color=clrs[cnt])

    axs[1].plot(inlier[1][0], inlier[1][1], 'o', markersize=2, color="tab:red")
    axs[1].plot([inlier[0][0], inlier[1][0]], [inlier[0][1], inlier[1][1]], color="tab:red", lw=1)

    if i % 200 == 0 and cnt < len(clrs):
        l2 = F @ ut.e2p(inlier[0])
        y = -(l2[0] / l2[1]) * x - (l2[2] / l2[1])
        axs[1].plot(x, y, color=clrs[cnt])
        cnt += 1

# for inlier_b in inliers_Hb:
#     plt.plot(inlier_b[0][0], inlier_b[0][1], 'o', markersize=2, color="tab:red")
#     plt.plot([inlier_b[0][0],inlier_b[1][0]], [inlier_b[0][1], inlier_b[1][1]], color="tab:red", lw=2)

# for outlier in outliers:
#     plt.plot(outlier[0][0], outlier[0][1], 'o', markersize=2, color="black")
#     plt.plot([outlier[0][0],outlier[1][0]], [outlier[0][1], outlier[1][1]], color="black", lw=0.5)

plt.xlim(0, 2816)
plt.ylim(0, 1880)
plt.gca().invert_yaxis()

plt.show()

# with a help of this well-written paper 'An Efficient Solution to the Five-Point Relative Pose Problem' by 'David Nister'
# shorturl.at/ekzHZ

# nice walkthrough
# https://stackoverflow.com/questions/22807039/decomposition-of-essential-matrix-validation-of-the-four-possible-solutions-for/22808118








