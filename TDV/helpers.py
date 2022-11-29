import numpy as np


def e2p(x: np.ndarray, alpha=1):
    return alpha * np.array([*x, 1])


def p2e(x: np.ndarray):
    el3 = x[-1]
    return x[:-1] / el3 if el3 != 0 else x[:-1] * float("inf")


def vlen(X):
    return np.apply_along_axis(func1d=np.linalg.norm, axis=0, arr=X)


def sqc(x):
    return np.array([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]])


def EuroRt(E, u1, u2):
    U, S, VT = np.linalg.svd(E)
    t1 = VT[3, :]
    t2 = -VT[3, :]
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    WT = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
    R1 = U @ W @ VT
    R2 = U @ WT @ VT
    Rs = [R1, R2]
    ts = [t1, t2]
    P0 = np.hstack([np.eyes(3), np.zeros((3,1))])
    Ps = [np.hstack([Rs[i], ts[k]]) for i in range(2) for k in range(2)]
    for P in Ps:
        if P @ :
    # CHIRALITY CONSTRAINT??????????
    return R, t


# mine:


def arr_3d_to_2d(arr: np.ndarray):
    return np.apply_along_axis(func1d=p2e_3d_to_2d, axis=0, arr=arr)


def arr_2d_to_3d(arr: np.ndarray):
    return np.apply_along_axis(func1d=e2p_2d_to_3d, axis=0, arr=arr)


