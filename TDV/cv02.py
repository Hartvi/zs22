import numpy as np
import math

npr = np.array
from matplotlib import pyplot as plt


K = npr([[1000, 0, 500],
         [0, 1000, 500],
         [0, 0, 1]])


def add_row(x: np.ndarray):
    shp = (1, x.shape[1])
    # print("shape1", shp)
    # print("new arr: ", np.concatenate((x, np.ones(shp) ), axis=0))
    return np.concatenate((x, np.ones(shp)), axis=0)


def p2e_3d_to_2d(x: np.ndarray):
    el3 = x[2]
    # print(el3)
    return x[:2] / el3 if el3 != 0 else x[:2]*float("inf")


def verticalize(x: np.ndarray or list):
    return npr([x]).transpose()


def create_I_minusC(global_camera_pos):
    return np.concatenate((np.eye(3), -verticalize(global_camera_pos)), axis=1)


def create_projection_matrix(R, t):
    return K @ R @ create_I_minusC(t)



"""
K, C 3x1, R 3x3
P = KR [I, -C] = K [R, -RC]
X 3x1 => u 2x1
 = P [X;1]
"""

P0 = np.concatenate((np.eye(3), np.zeros((3, 1))), axis=1)


def Xw_to_Xc(R: np.ndarray, t: np.ndarray, Xw):
    return R @ Xw + t


def world_to_camera_matrix_P(f, u0, v0):
    return npr([[f, 0, u0, 0], [0, f, v0, 0], [0, 0, 1, 0]])


def __(x: np.ndarray):
    return np.concatenate((x, [1]), axis=0)


def P_from_K(K):
    P = K @ P0


def Rz(x: float):
    return npr([[math.cos(x), -math.sin(x), 0], [math.sin(x), math.cos(x), 0], [0, 0, 1]])


def Ry(x: float):
    return npr([[math.cos(x), 0, math.sin(x)], [0, 1, 0], [-math.sin(x), 0, math.cos(x)]])


def Rx(x: float):
    return npr([[1, 0, 0], [0, math.cos(x), -math.sin(x)], [0, math.sin(x), math.cos(x)]])


def R(euler_angles: np.ndarray):
    return Rz(euler_angles[1]) @ Ry(euler_angles[1]) @ Rz(euler_angles[0])


if __name__ == "__main__":


    X1 = [[-0.5, 0.5, 0.5, -0.5, -0.5, -0.3, -0.3, -0.2, -0.2, 0, 0.5],
          [-0.5, -0.5, 0.5, 0.5, -0.5, -0.7, -0.9, -0.9, -0.8, -1, -0.5],
          [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]]
    X1 = npr(X1)

    X2 = [[-0.5, 0.5, 0.5, -0.5, -0.5, -0.3, -0.3, -0.2, -0.2, 0, 0.5],
          [-0.5, -0.5, 0.5, 0.5, -0.5, -0.7, -0.9, -0.9, -0.8, -1, -0.5],
          [4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5]]
    X2 = npr(X2)

    # using KR[I -C]
    I = np.eye(3)
    # zero rotation i guess? by default Z is forwards, y is down and x points right
    P1 = [[0,0,0], I]
    P2 = [[0, -1, 0], I]
    P3 = [[0, 0.5, 0], I]
    P4 = [[0, -3, 0.5], Rx(0.5)]
    P5 = [[0, -5, 4.2], Rx(np.pi * 0.5)]
    P6 = [[-1.5, -3, 1.5], Rx(0.8) @ Ry(-0.5)]
    tRs = [P1, P2, P3, P4, P5, P6]
    projection_matrices = []
    for tR in tRs:
        KRImC = create_projection_matrix(tR[1], tR[0])
        projection_matrices.append(KRImC)

    X1s = []
    X2s = []

    X1s2d = []
    X2s2d = []
    for i in range(len(projection_matrices)):
        x1res = projection_matrices[i] @ add_row(X1)
        x2res = projection_matrices[i] @  add_row(X2)
        x12dres = np.apply_along_axis(func1d=p2e_3d_to_2d, axis=0, arr=x1res)
        x22dres = np.apply_along_axis(func1d=p2e_3d_to_2d, axis=0, arr=x2res)
        X1s2d.append(x12dres)
        X2s2d.append(x22dres)
        X1s.append(x1res)
        X2s.append(x2res)

    # print(len(X1s2d))
    # print(len(X1s2d[0]))
    for i in range(6):
        plt.subplot(2, 3, i+1)
        u1 = X1s2d[i][0, :]
        v1 = X1s2d[i][1, :]
        u2 = X2s2d[i][0, :]
        v2 = X2s2d[i][1, :]
        # plt.subplot()
        plt.plot(u1, v1, 'r-', linewidth=2)
        plt.plot(u2, v2, 'b-', linewidth=2)
        plt.plot([u1, u2], [v1, v2], 'k-', linewidth=2)
        plt.gca().invert_yaxis()
        plt.axis('equal')  # this kind of plots should be isotropic
    plt.show()



