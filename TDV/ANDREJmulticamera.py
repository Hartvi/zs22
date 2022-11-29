
import os
from re import U
import time

from numpy.lib.type_check import real
import utils as ut
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
# from numpy.core.numeric import indices
from numpy.core.defchararray import replace
from numpy.linalg import det, lstsq, norm, inv
from scipy.spatial.transform import Rotation
from scipy.optimize import fmin

import json

from corresp import corresp
from geom_export import ge
from p5 import p5
from p3p import p3p
np.set_printoptions(suppress=True)

# ---- USEFUL DEBUGGING TOOLS ----
from icecream import ic          # call `ic()` in a function for orientation of where the code is now (instead of calling print("HERE")),
                                 # use `ic(thing)` instead of print("thing: ", thing)

import snoop                     # add `@snoop above` a function to pretty print every step of it
from loguru import logger        # add `@logger.catch` above a function to pretty print errors and the values that caused them
# import heartrate; heartrate.trace(browser=True)

def get_F_with_p5p(u1, u2, m12, n_it, eps, K, vb=True):
    """
    Args
    ----
    - `u1`  - [3 x n - homo]: left image corresponding points  
    - `u2`  - [3 x n - homo]: right image corresponding points  
    - `m12` - [n x 2 - idx]: correspondence map
    - `n_it`- [int]: number of iterations for RANSAC  
    - `eps` - [float32]: threshold value epsilon for RANSAC criterium 
    - `vb`  - [bool]: turns on printing (verbosity)

    Returns
    -------
    - `n_inliers` - [int]: returns the number of inliers fitting the criteria
    - `inliers`   - [idk yet]
    - `F`         - [3 x 3]: returns the best fitting fundamental matrix
    """

    # K undone points for Essential Matrix
    u1_K_undone = ut.e2p(u1.T)
    u1_K_undone = inv(K) @ u1_K_undone
    u1_K_undone = ut.p2e(u1_K_undone).T

    u2_K_undone = ut.e2p(u2.T)
    u2_K_undone = inv(K) @ u2_K_undone
    u2_K_undone = ut.p2e(u2_K_undone).T
    
    rng = np.random.default_rng()
    top_n_inliers = 0
    P2 = None
    maybe_P2 = None
    for i in range(n_it):
        five_indices = rng.choice(m12, 5, replace=False)
        u1p = np.zeros([2,5])
        u2p = np.zeros([2,5])
        
        # construct 5 and 5 points corresponding points for p5gb
        for n, idx in enumerate(five_indices):
            u1p[:,n] = u1_K_undone[idx[0]]
            u2p[:,n] = u2_K_undone[idx[1]]
        
        u1p = np.vstack([u1p, np.ones([1,5])])
        u2p = np.vstack([u2p, np.ones([1,5])])
        np.set_printoptions(suppress=True)
        
        # get maybe E matrices from external p5 algorithm
        maybe_Es = p5.p5gb(u1p, u2p)
        
        if vb:
            print("The number of possible Es: ", len(maybe_Es))
        for maybe_E in maybe_Es:
            U, D, V_trans = np.linalg.svd(maybe_E)
            if abs(D[0] - D[1]) > 0.01 or abs(D[2]) > 0.0001:
                print("Diagonal D is not correct: ", D)
                continue
        
            if det(U) < 0:
                U = -U
            if det(V_trans.T) < 0:
                V_trans = - V_trans

            W = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
            Ra = U @ W @ V_trans 
            Rb = U @ W.T @ V_trans 
            ta = U[:,-1].reshape(-1,1)  # third column of U
            
            P1 = np.hstack([np.eye(3), np.zeros([3,1])])
            # four possible solutions for the second camera
            P2s = [np.hstack([Ra, ta]), np.hstack([Ra, -ta]), np.hstack([Rb, ta]), np.hstack([Rb, -ta])]
            
            # now check cheirality constraint
            
            for P2_ in P2s:
                maybe_P2 = None
                valid_P = True
                for idx in range(np.shape(u1p)[1]): 
                    row_1 = u1p[:,idx].reshape(-1,1)[0] * P1[2,:]  - P1[0,:]
                    row_2 = u1p[:,idx].reshape(-1,1)[1] * P1[2,:]  - P1[1,:]
                    row_3 = u2p[:,idx].reshape(-1,1)[0] * P2_[2,:] - P2_[0,:]
                    row_4 = u2p[:,idx].reshape(-1,1)[1] * P2_[2,:] - P2_[1,:]

                    D = np.vstack([row_1, row_2, row_3, row_4])
                    Q = D.T @ D
                    U_q, D_q, V_q_trans = np.linalg.svd(Q)
                    X = U_q[:,-1]  # the last column of U is triangulated real-life point
                    X = ut.e2p(ut.p2e(X))
                    
                    X_P1 = P1 @ X
                    X_P2 = P2_ @ X

                    if X_P1[2] < 0 or X_P2[2] < 0:
                        valid_P = False
                        break
                
                if valid_P is True:
                    maybe_P2 = P2_
                    break
                
            if valid_P is False:
                # if vb:
                    # print("BADBADBADBADBABDAD ")
                break
                
            maybe_F = inv(K).T @ maybe_E @ inv(K)
            threshold = eps

            maybe_inliers = []
            maybe_indices = []
            for j, indices in enumerate(m12):
                u1_i = ut.e2p(u1[indices[0]])
                u2_i = ut.e2p(u2[indices[1]])

                # epipolar line on the left pic
                l1 = maybe_F.T @ u2_i

                # epipolar line on the right pic
                l2 = maybe_F @ u1_i

                # distance of u2 from epipolar line l2
                e_l = abs((l2[0] * u2_i[0] + l2[1] * u2_i[1] + l2[2])) / (np.sqrt(l2[0]**2 + l2[1]**2))
                
                # distance of u1 from eipolar line l1
                e_r = abs((l1[0] * u1_i[0] + l1[1] * u1_i[1] + l1[2])) / (np.sqrt(l1[0]**2 + l1[1]**2))
                
                avg_e = (e_l + e_r)/2

                if avg_e <= threshold:
                    maybe_inliers.append((ut.p2e(u1_i), ut.p2e(u2_i))) 
                    maybe_indices.append(j)
            
            if len(maybe_inliers) >= top_n_inliers:
                
                top_n_inliers = len(maybe_inliers)
                inliers = maybe_inliers
                best_indices = maybe_indices
                F = maybe_F
                P2 = maybe_P2

    return top_n_inliers, best_indices, inliers, F, P2


def fast_F_with_p5p(u1, u2, m12, n_it, eps, K, vb=True):
    """
    Args
    ----
    - `u1`  - [3 x n - homo]: left image corresponding points  
    - `u2`  - [3 x n - homo]: right image corresponding points  
    - `m12` - [n x 2 - idx]: correspondence map
    - `n_it`- [int]: number of iterations for RANSAC  
    - `eps` - [float32]: threshold value epsilon for RANSAC criterium 
    - `vb`  - [bool]: turns on printing (verbosity)

    Returns
    -------
    - `n_inliers` - [int]: returns the number of inliers fitting the criteria
    - `indices`   - [idk yet]
    - `F`         - [3 x 3]: returns the best fitting fundamental matrix
    - `P2`        - [4 x 3]: returns the best fitting camera matrix in format [R|t] (not K[R|t])
    """

    # K undone points for Essential Matrix
    u1_K_undone = ut.p2e(inv(K) @ u1)
    u2_K_undone = ut.p2e(inv(K) @ u2)
    
    rng = np.random.default_rng()
    top_n_inliers = 0
    P2 = None
    maybe_P2 = None
    for i in range(n_it):
        five_indices = rng.choice(m12, 5, replace=False)
        u1p = np.zeros([2,5])
        u2p = np.zeros([2,5])
        
        # construct 5 and 5 points corresponding points for p5gb
        for n, idx in enumerate(five_indices):
            u1p[:,n] = u1_K_undone[:, idx[0]]
            u2p[:,n] = u2_K_undone[:, idx[1]]
        
        u1p = np.vstack([u1p, np.ones([1,5])])
        u2p = np.vstack([u2p, np.ones([1,5])])
        np.set_printoptions(suppress=True)
        
        # get maybe E matrices from external p5 algorithm
        maybe_Es = p5.p5gb(u1p, u2p)
        
        if vb:
            print("The number of possible Es: ", len(maybe_Es))
        for maybe_E in maybe_Es:
            U, D, V_trans = np.linalg.svd(maybe_E)
            if abs(D[0] - D[1]) > 0.01 or abs(D[2]) > 0.0001:
                print("Diagonal D is not correct: ", D)
                continue
        
            if det(U) < 0:
                U = -U
            if det(V_trans.T) < 0:
                V_trans = - V_trans

            W = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
            Ra = U @ W @ V_trans 
            Rb = U @ W.T @ V_trans 
            ta = U[:,-1].reshape(-1,1)  # third column of U
            
            P1 = np.hstack([np.eye(3), np.zeros([3,1])])
            # four possible solutions for the second camera
            P2s = [np.hstack([Ra, ta]), np.hstack([Ra, -ta]), np.hstack([Rb, ta]), np.hstack([Rb, -ta])]
            
            # now check cheirality constraint
            # this is done for 5 points only, for cycle is ok hehe
            for P2_ in P2s:
                maybe_P2 = None
                valid_P = True
                for idx in range(np.shape(u1p)[1]): 
                    row_1 = u1p[:,idx].reshape(-1,1)[0] * P1[2,:]  - P1[0,:]
                    row_2 = u1p[:,idx].reshape(-1,1)[1] * P1[2,:]  - P1[1,:]
                    row_3 = u2p[:,idx].reshape(-1,1)[0] * P2_[2,:] - P2_[0,:]
                    row_4 = u2p[:,idx].reshape(-1,1)[1] * P2_[2,:] - P2_[1,:]

                    D = np.vstack([row_1, row_2, row_3, row_4])
                    Q = D.T @ D
                    U_q, D_q, V_q_trans = np.linalg.svd(Q)
                    X = U_q[:,-1]  # the last column of U is triangulated real-life point
                    X = ut.e2p(ut.p2e(X))
                    
                    X_P1 = P1 @ X
                    X_P2 = P2_ @ X

                    if X_P1[2] < 0 or X_P2[2] < 0:
                        valid_P = False
                        break
                
                if valid_P is True:
                    maybe_P2 = P2_
                    break
                
            if valid_P is False:
                break
                
            maybe_F = inv(K).T @ maybe_E @ inv(K)
            u1_sorted = np.take(u1, m12[:,0], axis=1)
            u2_sorted = np.take(u2, m12[:,1], axis=1)

            l1s = maybe_F.T @ u2_sorted
            l2s = maybe_F @ u1_sorted

            e_ls = abs(l2s[0,:] * u2_sorted[0,:] + l2s[1,:] * u2_sorted[1,:] + l2s[2,:]) / np.sqrt(l2s[0,:]**2 + l2s[1,:] **2)
            e_rs = abs(l1s[0,:] * u1_sorted[0,:] + l1s[1,:] * u1_sorted[1,:] + l1s[2,:]) / np.sqrt(l1s[0,:]**2 + l1s[1,:] **2)

            avg_es = (e_ls + e_rs) / 2
            maybe_indices = np.where(avg_es < eps)[0]
            maybe_inlier_m12 = np.take(m12, maybe_indices, axis=0)
            
            if len(maybe_inlier_m12) >= top_n_inliers:
                top_n_inliers = len(maybe_inlier_m12)
                inliers = maybe_inlier_m12
                best_indices = maybe_indices
                
                F = maybe_F
                P2 = maybe_P2

    return top_n_inliers, inliers, best_indices, F, P2


def triangulate_w_SVD(u1p, u2p, m12, P1, P2, K, F=None):

    # for_E = np.vstack([P2, np.array([0,0,0,1])]) @ inv(np.vstack([P1, np.array([0,0,0,1])]))
    # E = ut.sqc(for_E[0:3,3]) @ for_E[0:3,0:3]  
    # F_comp = inv(K).T @ E @ inv(K)
    # ic(F)
    # ic(F_comp)

    u1 = ut.e2p(ut.p2e(inv(K) @ u1p))  # K undone
    u2 = ut.e2p(ut.p2e(inv(K) @ u2p))

    if F is not None:
        P1 = K @ P1  # the matrices P were created as K undone
        P2 = K @ P2
        u1 = u1p
        u2 = u2p  

    # now working in pixels in also Ps and u_i
    X = np.array(np.zeros([3,np.shape(m12)[0]]))
    for idx, indices in enumerate(m12):
        
        xp_i = u1[:, indices[0]].reshape(-1,1)
        yp_i = u2[:, indices[1]].reshape(-1,1)

        # first correct the points with sampson's reprojection error
        # ---- SAMPSONS ERROR CORRECTION start ----
        if F is not None:
            epsilon = yp_i.T @ F @ xp_i
            S = np.array([[1, 0, 0], [0, 1, 0]])
            j_row1 = S @ F.T @ yp_i
            j_row2 = S @ F @ xp_i
            
            J = np.vstack([j_row1, j_row2]).T
            stacked_pnts = np.array([xp_i[0],xp_i[1],yp_i[0],yp_i[1]])
            new_stacked_pnts = stacked_pnts - (epsilon/norm(J)**2) * J.T
            
            xp_i = new_stacked_pnts[0:2]
            yp_i = new_stacked_pnts[2:4]

        # ---- SAMPSONS ERROR CORRECTION end ----

        row_1 = xp_i[0] * P1[2,:]  - P1[0,:]
        row_2 = xp_i[1] * P1[2,:]  - P1[1,:]
        row_3 = yp_i[0] * P2[2,:] - P2[0,:]
        row_4 = yp_i[1] * P2[2,:] - P2[1,:]
    
        D = np.vstack([row_1, row_2, row_3, row_4])
        Q = D.T @ D

        if F is not None:
            # D may be ill conditioned for numerical computation, which results
            # in a poor estimate of X_i
            S = np.diag(1/np.max(np.abs(D), 1))
            D_dash = D @ S 
            Q = D_dash.T @ D_dash
        U_q, D_q, V_q_trans = np.linalg.svd(Q)
        X_i = U_q[:,-1]  # the last column of U is triangulated real-life point

        if F is not None:
            X_i = S @ X_i  # undo the rescale the numerical conditioning matrix
        X_i = ut.p2e(X_i)
        X[:, idx] = X_i.flatten()
    return X


def get_Rt_with_p3p(map_Xu, X, u_pixels, n_it, eps, K, vb=True):
    """
    Args
    ----
    - map_Xu: [n x 2] map of X indices (left column) to u indices (right column)

    """

    # up points need to be K undone first
    up_K_undone = ut.e2p(ut.p2e(inv(K) @ ut.e2p(u_pixels)))
    rng = np.random.default_rng()

    best_n_inliers = 0
    best_inlier_indices = None
    R = None
    t = None

    sorted_X = np.take(X, map_Xu[:,0], axis=1)
    sorted_u_pixels = np.take(u_pixels, map_Xu[:,1], axis=1)

    # iteration of RANSAC
    # TODO adaptive number of iterations
    for it in range(n_it):
        maybe_inlier_indices = None
        three_indices = rng.choice(map_Xu, 3, replace=False)

        X_3 = np.zeros([4,3])
        u_3 = np.zeros([3,3])
        for i, real_idx in enumerate(three_indices):
            X_3[:, i] = ut.e2p(X[:, real_idx[0]]).flatten()
            u_3[:, i] = up_K_undone[:, real_idx[1]]

        X_c = p3p.p3p_grunert(X_3, u_3)
        for maybe_X in X_c:
            _R,_t = p3p.XX2Rt_simple(X_3, maybe_X)
            _P = K @ np.hstack([_R,_t])

            temp_X = np.hstack([_R,_t]) @ ut.e2p(sorted_X)
            indices_infront_camera = np.where(temp_X[2,:] > 0)[0]

            X_p = np.take(sorted_X, indices_infront_camera, axis=1)
            temp_map_Xu = np.take(map_Xu, indices_infront_camera, axis=0)
            subset_of_sorted_u_pixels = np.take(sorted_u_pixels, indices_infront_camera, axis=1)
            
            u_proj = ut.p2e( _P @ ut.e2p(X_p))
            
            dist_arr = norm(u_proj - subset_of_sorted_u_pixels, axis=0)

            # TODO maybe do ML estimator instead of thresholding
            maybe_inlier_indices = np.where(dist_arr < eps)[0] 
            inlier_map_Xu = np.take(temp_map_Xu, maybe_inlier_indices, axis=0)

            # TODO check if the inlier_map_Xu magic is correct

            if len(maybe_inlier_indices) > best_n_inliers: # TODO is len ok here?
                best_n_inliers = len(maybe_inlier_indices)
                R = _R
                t = _t
                best_inlier_indices = maybe_inlier_indices
                best_map = inlier_map_Xu
    
    return R, t, best_map, best_inlier_indices


def better_get_Rt_with_p3p(map_Xu, X, u_pixels, n_it, eps, K, vb=True):
    """
    Args
    ----
    - map_Xu: [n x 2] map of X indices (left column) to u indices (right column)

    """
    # X[:, Xu[:, 0]], u[:, Xu[:, 1]]
    # up points need to be K undone first
    up_K_undone = ut.e2p(ut.p2e(inv(K) @ ut.e2p(u_pixels)))
    rng = np.random.default_rng()

    best_n_inliers = 0
    best_inlier_indices = None
    R = None
    t = None

    # sorted_X = np.take(X, map_Xu[:,0], axis=1)
    # sorted_u_pixels = np.take(u_pixels, map_Xu[:,1], axis=1)

    

    # iteration of RANSAC
    # TODO adaptive number of iterations
    for it in range(n_it):
        maybe_inlier_indices = None
        three_indices = rng.choice(map_Xu, 3, replace=False)

        X_3 = np.zeros([4,3])
        u_3 = np.zeros([3,3])
        for i, real_idx in enumerate(three_indices):
            X_3[:, i] = ut.e2p(X[:, real_idx[0]]).flatten()
            u_3[:, i] = up_K_undone[:, real_idx[1]]

        X_c = p3p.p3p_grunert(X_3, u_3)
        for maybe_X in X_c:
            _R,_t = p3p.XX2Rt_simple(X_3, maybe_X)
            _P = K @ np.hstack([_R,_t])


            # ------------- HERE -------------
            # X[:, Xu[:, 0]], u[:, Xu[:, 1]]
            sorted_X = X[:, map_Xu[:,0]]

            temp_X = np.hstack([_R,_t]) @ ut.e2p(sorted_X)
            indices_infront_camera = np.where(temp_X[2,:] > 0)[0]

            # X_p = np.take(sorted_X, indices_infront_camera, axis=1)
            X_p = sorted_X[:, indices_infront_camera]
            # temp_map_Xu = np.take(map_Xu, indices_infront_camera, axis=0)
            temp_map_Xu = map_Xu[indices_infront_camera, :]
            # subset_of_sorted_u_pixels = np.take(sorted_u_pixels, indices_infront_camera, axis=1)
            subset_of_sorted_u_pixels = u_pixels[:, map_Xu[:, 1]][:, indices_infront_camera]
            
            u_proj = ut.p2e( _P @ ut.e2p(X_p))
            
            dist_arr = norm(u_proj - subset_of_sorted_u_pixels, axis=0)

            # TODO maybe do ML estimator instead of thresholding
            maybe_inlier_indices = np.where(dist_arr < eps)[0] 
            inlier_map_Xu = np.take(temp_map_Xu, maybe_inlier_indices, axis=0)

            # TODO check if the inlier_map_Xu magic is correct

            if len(maybe_inlier_indices) > best_n_inliers: # TODO is len ok here?
                best_n_inliers = len(maybe_inlier_indices)
                R = _R
                t = _t
                best_inlier_indices = maybe_inlier_indices
                best_map = inlier_map_Xu
    
    return R, t, best_map, best_inlier_indices


def get_pnts(data_path, cam_num):
    path = data_path + '/' + 'u_' + '{:02}'.format(cam_num) + '.txt'
    return np.loadtxt(path).T

# @logger.catch
def optim_func(x, K, Xs, u_pixels, R_0):
    R_r = Rotation.from_rotvec(x[0:3]).as_matrix()
    R = R_r @ R_0
    t = x[3:6].reshape(-1,1)
    X_projected = ut.p2e(K @ np.hstack([R,t]) @ Xs)
    # ic((X_projected - u_pixels)**2)
    # ic(np.sum((X_projected - u_pixels)**2, axis=1))
    # ic(np.sqrt(np.sum((X_projected - u_pixels)**2, axis=0)))
    # ic(np.mean(np.sqrt(np.sum((X_projected - u_pixels)**2))))

    sqr_mean_err = np.mean(np.sqrt(np.sum((X_projected - u_pixels)**2, axis=0)))
    # ic(sqr_mean_err)
    return sqr_mean_err


T_START = time.time()

path_to_imgs  = 'imgs'
path_to_corrs = 'data/corrs'
path_to_points = 'data/points'

img_names = [path_to_imgs + '/' + file for file in os.listdir(path_to_imgs)]
corr_names_short = [file for file in os.listdir(path_to_corrs)]
corr_names = [path_to_corrs + '/' + file for file in os.listdir(path_to_corrs)]
pnt_names = [path_to_points + '/' + file for file in os.listdir(path_to_points)]

img_names.sort()
corr_names.sort()
corr_names_short.sort()
pnt_names.sort()

K = np.array([[2080,  0,     1421],
              [0,     2080,  957],
              [0,     0,     1]])
       
all_points = dict()
for name in pnt_names:
    num = name[-6:-4]  # the number part from the points name eg. num = '01' from name = 'u_01'
    all_points[num] = np.loadtxt(name).T

# ---- CAMERAS SETUP ----
unused_cams = [_ for _ in range(12)]
Ps = [None for _ in range(12)]


# ---- CREATE BODY OF CAMS AND THEIR CORRESPONDENCIES ----

c = corresp.Corresp(12)
c.verbose = 1
combos = []
for combination in corr_names_short:
    comb_i = combination[2:4]
    comb_j = combination[5:7]
    pnts_i = all_points[comb_i]
    pnts_j = all_points[comb_j]
    indices_ij = np.loadtxt(path_to_corrs + '/m_' + comb_i + '_' + comb_j + '.txt', dtype='int')
    c.add_pair( int(comb_i), int(comb_j), indices_ij)
    combos.append((int(comb_i), int(comb_j)))

# start with the easiest picture 5 and 6 --> in python coords 4 and 5

Ps[4] = np.array(np.hstack([np.eye(3), np.zeros([3,1])]))

m1, m2 = c.get_m(4,5)
m12 = np.vstack((m1,m2)).T
u1 = np.loadtxt(path_to_points + '/' + "u_04.txt").T
u2 = np.loadtxt(path_to_points + '/' + "u_05.txt").T

# image to image, get inliers
n_in, inliers, map_indices, F, P2 = fast_F_with_p5p(ut.e2p(u1), ut.e2p(u2), m12, 100, 3, K, False)
ic(n_in)
Ps[5] = P2

X = triangulate_w_SVD(ut.e2p(u1), ut.e2p(u2), inliers, Ps[4], Ps[5], K)

c.start(4, 5, map_indices)  # propagation of correspondencies

while unused_cams != []:
    Xu_count = c.get_Xucount(c.get_green_cameras()[0])
    ic(c.get_green_cameras()[0])
    ic(c.get_Xucount(c.get_green_cameras()[0]))
    if(len(c.get_green_cameras()[0])) == 0:
        ic("Ending the gluing sequence, no Green cameras left.")
        break
    best_cam_ID = c.get_green_cameras()[0][np.argmax(Xu_count)]
    ic(best_cam_ID)


    Xu = c.get_Xu(best_cam_ID)
    u_i = get_pnts(path_to_points, best_cam_ID)
    map_Xu = np.array(Xu)[0:2, :].T
    
    R_0, t_0, map_Xu_i, map_Xu_indices = get_Rt_with_p3p(map_Xu, X, u_i, 200, 3, K)
    t_0 = t_0.flatten()
    ic(R_0)
    ic(t_0)
    # maybe problem with the indexing of X and u_i via inliers?
    x = fmin(optim_func, np.array([0,0,0,t_0[0],t_0[1],t_0[2]]), args=(K, ut.e2p(X[:, map_Xu_i[:,0]]), u_i[:, map_Xu_i[:,1]], R_0)) # good

    R_i = Rotation.from_rotvec(x[0:3]).as_matrix()
    t_i = x[3:6].reshape(-1,1)
    R_i = R_i @ R_0
    ic(R_i)
    ic(t_i)
    Ps[best_cam_ID] = np.hstack([R_i, t_i])

    
    unused_cams.remove(best_cam_ID)
    ic("Removed the camera with camID: ", best_cam_ID)
    c.join_camera(best_cam_ID, map_Xu_indices)
    ilist = c.get_cneighbours(best_cam_ID)
    ic(ilist)
    for cam_i in ilist:
        
        m_ij = np.array(c.get_m(best_cam_ID, cam_i)).T
        u_i = get_pnts(path_to_points, best_cam_ID)
        u_j = get_pnts(path_to_points, cam_i)    

        X_i = triangulate_w_SVD(ut.e2p(u_i), ut.e2p(u_j), m_ij, Ps[best_cam_ID], Ps[cam_i], K)


        # now test with reprojection
        cam_i_reprojection = norm(ut.p2e(K @ Ps[cam_i] @ ut.e2p(X_i)) - u_j[:, m_ij[:,1]], axis=0).reshape(-1,1) < 5
        best_cam_reprojection = norm(ut.p2e(K @ Ps[best_cam_ID] @ ut.e2p(X_i)) - u_i[:, m_ij[:,0]], axis=0).reshape(-1,1) < 5
        both_good = np.logical_and(cam_i_reprojection, best_cam_reprojection)
        reprojected_indices = np.where(both_good == True)[0]
        X_good = X_i[:,reprojected_indices]

        X = np.hstack([X, X_good])
        
        c.new_x(best_cam_ID, cam_i, reprojected_indices)
    
    ilist = c.get_selected_cameras()
    for ver_cam in ilist:
        xx, uu, Xu_verified = c.get_Xu(ver_cam)
        map_Xu = np.hstack([xx.reshape(-1,1), uu.reshape(-1,1)])
        Xu_tentative = np.where(Xu_verified == False)[0]


        if len(Xu_tentative) == 0:
            c.verify_x(ver_cam, [])
        else:
            ic("=========== Need to verify here! ===========")
            map = map_Xu[Xu_tentative, :]
            u_ver_cam = get_pnts(path_to_points, ver_cam)
            repro = norm(ut.p2e(K @ Ps[ver_cam] @ ut.e2p(X[:,map[:,0]])) - u_ver_cam[:, map[:,1]], axis=0).reshape(-1,1) < 5
            reprojected_indices = np.where(repro == True)[0]
            # ic(reprojected_indices)
            # ic(Xu_verified)
            c.verify_x(ver_cam, Xu_tentative[reprojected_indices])
        
    c.finalize_camera()

    # g = ge.GePly('last.ply')
    # g.points(X) # Xall contains euclidean points (3xn matrix), ColorAll RGB colors (3xn or 3x1, optional)
    # g.close()
        
g = ge.GePly('final_3px_correction.ply')
g.points(X) # Xall contains euclidean points (3xn matrix), ColorAll RGB colors (3xn or 3x1, optional)
g.close()      
  

# TODO how to visualize the cameras?
# visualize camera centers

np.save('cams.npy', Ps)

T_RUN = time.time() - T_START
print("Time running: ", T_RUN)
