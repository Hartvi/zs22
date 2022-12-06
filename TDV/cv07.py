import numpy as np
import os
import PIL.Image
import p3p
import math
import tools
from scipy.spatial.transform import Rotation
from scipy import optimize

import bundle_adjustment as ba
import cv06

npr = np.array
# p3p.p3p_grunert()


def eval_inliers(u1p, u2p, F, threshold):
    errs = tools.err_F_sampson(F, u1p, u2p)
    inlier_indices = cv06.get_inliers(errs, threshold)
    return inlier_indices


def isin_w_duplicates(x, y):
    ret1 = list()
    ret2 = list()
    # np.isin() for example doesnt work for x = [1,1,2] and y = [1,2] => sumy = 2, sumx = 3
    for k, i in enumerate(y):
        ret1 += np.where(i == x)[0].tolist()
        # if len(np.where(i == y)[0].tolist()) > 1:
        #     print(np.where(i == y)[0].tolist())
    for k, i in enumerate(x):
        ret2 += np.where(i == y)[0].tolist()
        # if len(np.where(i == x)[0].tolist()) > 1:
        #     print(np.where(i == x)[0].tolist())
    assert len(ret1) == len(ret2), "wrong isin_w_duplicates"
    return npr(ret1), npr(ret2)


def sort_ids(i1, i2) -> tuple:
    return tuple(sorted((i1, i2)))

def opt_p3p(R, inlier_Xs, inlier_us, eps2):

    def err_func(args):
        rot_vec = args[:3]
        new_t = args[3:6].reshape(-1,1)
        tmp_R = Rotation.from_rotvec(rotvec=rot_vec).as_matrix()
        new_R = tmp_R @ R
        tmp_P = np.hstack([new_R, new_t])
        temp_X = tmp_P @ inlier_Xs
        indices_infront_camera = np.where(temp_X[2, :] > 0)[0]

        X_p = inlier_Xs[:, indices_infront_camera]
        subset_of_sorted_u_pixels = tools.p2e(inlier_us[:, indices_infront_camera])

        u_proj = tools.p2e(cv06.K @ tmp_P @ X_p)

        # eval error
        err_elements = np.sum((u_proj - subset_of_sorted_u_pixels)**2, axis=0)
        support = np.sum(err_elements)
        return support
    
    return err_func


def get_Rt_with_p3p(Xu_idx, u_idx, X, u_pixels, n_it, eps):
    assert X.shape[0] == 3 or X.shape[0] == 4, "3d point have to be either 3d or 4d (hom) along columns"
    assert Xu_idx.shape == u_idx.shape, "X to u mapping have to have the same size, Xu_idx: "+str(Xu_idx.shape)+"  u_idx: "+str(u_idx.shape)
    assert u_pixels.shape[0] == 2 or u_pixels.shape[0] == 3, "pixel points have to be either 2d or 3d (hom)"

    # homogenize the 3d X and 2d u_pixels
    if X.shape[0] == 3:
        X = tools.e2p(X)
    if u_pixels.shape[0] == 2:
        u_pixels = tools.e2p(u_pixels)

    eps2 = eps**2
    number_of_matches = len(Xu_idx)
    # up points need to be K undone first
    up_K_undone = tools.e2p(tools.p2e(cv06.invK @ u_pixels))
    rng = np.random.default_rng(cv06.rng_seed)

    best_inlier_Xu_idx, best_inlier_u_idx = None, None
    best_n_inliers = 0
    best_support = -float('inf')
    best_inlier_indices = None
    R = None
    t = None

    hom_sel_X = X[:, Xu_idx]
    hom_sel_u_pixels = u_pixels[:, u_idx]

    # iteration of RANSAC
    it = 0
    while it < n_it:
        maybe_inlier_indices = None
        three_indices = rng.choice(number_of_matches, 3, replace=False)
        three_X_idx = Xu_idx[three_indices]
        three_u_idx = u_idx[three_indices]

        X_3 = X[:, three_X_idx]
        u_3 = up_K_undone[:, three_u_idx]

        X_c = p3p.p3p_grunert(X_3, u_3)
        for maybe_X in X_c:
            _R, _t = p3p.XX2Rt_simple(X_3, maybe_X)
            _P = cv06.K @ np.hstack([_R, _t])

            temp_X = np.hstack([_R, _t]) @ hom_sel_X
            indices_infront_camera = np.where(temp_X[2, :] > 0)[0]

            X_p = hom_sel_X[:, indices_infront_camera]
            tmp_Xu_idx = Xu_idx[indices_infront_camera]
            tmp_u_idx = u_idx[indices_infront_camera]
            subset_of_sorted_u_pixels = tools.p2e(hom_sel_u_pixels[:, indices_infront_camera])

            u_proj = tools.p2e(_P @ X_p)

            # eval error
            err_elements = np.sum((u_proj - subset_of_sorted_u_pixels)**2, axis=0)
            maybe_inlier_indices = (err_elements < eps2)
            errs = 1 - err_elements / eps2
            errs[errs < 0] = 0
            support = np.sum(errs)

            inlier_Xu_idx = tmp_Xu_idx[maybe_inlier_indices]
            inlier_u_idx = tmp_u_idx[maybe_inlier_indices]

            if support > best_support:

                n_it = tools.Nmax(0.80, np.sum(maybe_inlier_indices) / number_of_matches, 3)
                R = _R
                t = _t
                best_inlier_indices = maybe_inlier_indices
                best_inlier_Xu_idx = inlier_Xu_idx
                best_inlier_u_idx = inlier_u_idx
                best_support = support
                print("max_iters: ", n_it, " current iter:", it, "p3p best support: ", support)
        it += 1


    ig = np.zeros(6)
    ig[3:6] = t.flatten()
    xopt = optimize.fmin(func=opt_p3p(R, X[:, best_inlier_Xu_idx], u_pixels[:, best_inlier_u_idx], eps2), x0=ig)
    rot_vec = xopt[:3]
    _t = xopt[3:6].reshape((-1, 1))
    _R = Rotation.from_rotvec(rotvec=rot_vec).as_matrix() @ R

    _P = cv06.K @ np.hstack([_R, _t])

    temp_X = np.hstack([_R, _t]) @ hom_sel_X
    indices_infront_camera = np.where(temp_X[2, :] > 0)[0]

    X_p = hom_sel_X[:, indices_infront_camera]
    tmp_Xu_idx = Xu_idx[indices_infront_camera]
    tmp_u_idx = u_idx[indices_infront_camera]
    subset_of_sorted_u_pixels = tools.p2e(hom_sel_u_pixels[:, indices_infront_camera])

    u_proj = tools.p2e(_P @ X_p)

    # eval error
    err_elements = np.sum((u_proj - subset_of_sorted_u_pixels)**2, axis=0)
    maybe_inlier_indices = (err_elements < eps2)
    errs = 1 - err_elements / eps2
    errs[errs < 0] = 0
    support = np.sum(errs)

    inlier_Xu_idx = tmp_Xu_idx[maybe_inlier_indices]
    inlier_u_idx = tmp_u_idx[maybe_inlier_indices]

    if support > best_support:

        n_it = tools.Nmax(0.80, np.sum(maybe_inlier_indices) / number_of_matches, 3)
        R = _R
        t = _t
        best_inlier_indices = maybe_inlier_indices
        best_inlier_Xu_idx = inlier_Xu_idx
        best_inlier_u_idx = inlier_u_idx
        best_support = support

    return R, t, best_inlier_Xu_idx, best_inlier_u_idx, best_inlier_indices


class ScenePoint:
    scene_point_count = 0
    unique_ids = set()

    def __init__(self, space_coords, name: int):
        self.space_coords = space_coords
        assert name not in ScenePoint.unique_ids, "Trying to create point with already existing ID: " + str(name)
        self.name = name
        ScenePoint.unique_ids.add(name)
        ScenePoint.scene_point_count += 1


class CameraContainer:
    def __init__(self, num_of_cams, threshold, root_path="/home/hartvi/zs22/TDV/scene_1"):
        self.num_of_cams = min(num_of_cams, 12)
        # NEW VERSION
        self.threshold = threshold
        # pairwise
        self.ms = dict()
        self.P2s = dict()
        self.m_inlier_indices = dict()
        self.Fs = dict()
        # single
        self.us = dict()
        self.all_cameras = set()
        self.red_cameras = set()
        self.green_cameras = set()
        self.Xus = dict()  # cam_id => point name => u index
        # self.Xus[cam_id] = [[], []]  # Xus[0]: point name, Xus[1]: u name
        self.Xus_tentative_sps_idx = dict()  # cam_id => point name => u index
        self.Xus_tentative_u_idx = dict()  # cam_id => point name => u index
        # point name => 3d data
        self.scene_points = list()

        for i in range(num_of_cams):
            cam_id = i+1
            points = np.loadtxt(os.path.join(root_path, "corresp", "u_{:02d}.txt".format(cam_id)))
            u = tools.e2p(points.T)
            img = os.path.join(root_path, "images", "{:02d}.jpg".format(cam_id))
            # NEW VERSION
            self.us[cam_id] = u  # cam id => u pixels in K domain
            self.ms[cam_id] = dict()
            self.Fs[cam_id] = dict()
            self.m_inlier_indices[cam_id] = dict()
            self.Xus[cam_id] = [[], []]  # Xus[0]: point name, Xus[1]: u name
            self.Xus_tentative_sps_idx[cam_id] = list()  # list[3d point name]
            self.Xus_tentative_u_idx[cam_id] = list()  # list[u_index]
            self.all_cameras.add(cam_id)
        for i in range(num_of_cams):
            for k in range(i+1, num_of_cams):
                cam1_id = i+1
                cam2_id = k+1
                corresp = np.loadtxt(os.path.join(root_path, "corresp", "m_{:02d}_{:02d}.txt".format(cam1_id, cam2_id)), dtype=int).T
                self.ms[cam1_id][cam2_id] = corresp

    def get_ms(self, id1, id2):
        assert id1 != id2, "cam_ids mustnt equal: id1: "+str(id1)+" id2: "+str(id2)
        if id1 < id2:
            ms = self.ms[id1][id2]  # a=>b
            return ms[0], ms[1]
        else:
            ms = self.ms[id2][id1]  # reverse order, b=>a
            return ms[1], ms[0]

    def set_ms(self, id1, id2, m1, m2):
        if id1 < id2:
            self.ms[id1][id2] = np.array([m1, m2])
        else:
            self.ms[id2][id1] = np.array([m2, m1])

    def initialize(self, start_cam1, start_cam2):
        pass
        start_cam1, start_cam2 = sort_ids(start_cam1, start_cam2)
        m1, m2 = self.get_ms(start_cam1, start_cam2)
        u1all = self.us[start_cam1]
        u2all = self.us[start_cam2]
        u1 = u1all[:, m1]
        u2 = u2all[:, m2]
        F, P2, m_inlier_indices, X = cv06.ransac(u1, u2, threshold=self.threshold)
        assert m_inlier_indices.shape[0] == X.shape[1], "Num of 3d points X should equal number of inliers. X: "+str(X.shape)+" m_inlier_indices: "+str(m_inlier_indices.shape)
        # print("m_inlier_indices:", m_inlier_indices)
        self.Fs[start_cam1][start_cam2] = F
        self.P2s[start_cam1] = np.eye(3, 4)
        self.P2s[start_cam2] = P2
        self.m_inlier_indices[start_cam1][start_cam2] = m_inlier_indices

        u1_inlier_indices = m1[m_inlier_indices]
        u2_inlier_indices = m2[m_inlier_indices]
        for i in range(m_inlier_indices.shape[0]):
            # TODO: cross check 3D point whether its new???
            Xu_name = i
            self.Xus[start_cam1][0].append(Xu_name)
            self.Xus[start_cam1][1].append(u1_inlier_indices[i])
            self.Xus[start_cam2][0].append(Xu_name)
            self.Xus[start_cam2][1].append(u2_inlier_indices[i])
            # self.Xus[start_cam2][Xu_name] = u2_inlier_indices[i]
            self.scene_points.append(X[:, i])

        self.set_ms(start_cam1, start_cam2, npr([]), npr([]))
        self.red_cameras |= {start_cam1, start_cam2}
        return start_cam1, start_cam2, m_inlier_indices, list(range(m_inlier_indices.shape[0]))

    def start(self, i1, i2, m12_inlier_idx, scene_point_names):
        i12 = sort_ids(i1, i2)
        m12_sp_names = {i1: npr(self.Xus[i1][0]), i2: npr(self.Xus[i2][0])}
        m12_inliers = {i1: npr(self.Xus[i1][1]), i2: npr(self.Xus[i2][1])}
        for cam_i in i12:
            for cam_k in range(1, self.num_of_cams+1):
                if cam_k == cam_i:
                    continue
                if cam_i == i1 and cam_k == i2:
                    continue
                if cam_k == i1 and cam_i == i2:
                    continue
                mi, mk = self.get_ms(cam_i, cam_k)

                mk_idx_to_mi, original_inliers_to_cam_k = isin_w_duplicates(m12_inliers[cam_i], mi)  # can have duplicate scene to screen correspondences
                # print("mk_idx_to_mi", mk_idx_to_mi.shape, np.max(mk_idx_to_mi))
                # print("m12_sp_names[cam_i]", m12_sp_names[cam_i].shape)
                # print()
                # print("original_inliers_to_cam_k", original_inliers_to_cam_k.shape, np.max(original_inliers_to_cam_k))
                # print("mk", mk.shape)
                self.Xus_tentative_sps_idx[cam_k] += m12_sp_names[cam_i][mk_idx_to_mi].tolist()
                self.Xus_tentative_u_idx[cam_k] += mk[original_inliers_to_cam_k].tolist()
                assert len(self.Xus_tentative_sps_idx[cam_k]) == len(self.Xus_tentative_u_idx[cam_k]), "handle duplicates"
                # print("self.Xus_tentative_sps_idx[cam_k]", len(self.Xus_tentative_sps_idx[cam_k]))
                # print("self.Xus_tentative_u_idx[cam_k]", len(self.Xus_tentative_u_idx[cam_k]))
                tmp_ms = self.get_ms(cam_i, cam_k)
                # remove the added correspondences
                tmpm1, tmpm2 = npr(tmp_ms)[:, ~original_inliers_to_cam_k]
                self.set_ms(cam_i, cam_k, tmpm1, tmpm2)

        for tentative_cam_id in self.Xus_tentative_sps_idx:
            if len(self.Xus_tentative_sps_idx[tentative_cam_id]) > 0:
                self.green_cameras.add(tentative_cam_id)

    def attach_camera(self):
        assert len(self.red_cameras) < self.num_of_cams, "trying to add "+str(len(self.red_cameras)+1)+"th camera. Only "+str(self.num_of_cams)+" cams available though."
        cams = np.zeros(len(self.Xus_tentative_sps_idx), dtype=int)
        tentatives = np.zeros(len(self.Xus_tentative_sps_idx), dtype=int)
        for k, cam_id in enumerate(self.Xus_tentative_sps_idx):
            cams[k] = cam_id
            tentatives[k] = len(self.Xus_tentative_u_idx[cam_id])

        best_tentative_green_cam_id = cams[np.argsort(tentatives)[-1]]
        assert best_tentative_green_cam_id in self.green_cameras, "tentative correspondences must belong to green camera. This one isnt in it: "+str(best_tentative_green_cam_id)
        print("best cam:", best_tentative_green_cam_id)
        u_idx = self.Xus_tentative_u_idx[best_tentative_green_cam_id]
        u_idx = npr(u_idx)
        X_names = self.Xus_tentative_sps_idx[best_tentative_green_cam_id]
        X_names = npr(list(X_names))  # X's found in the green camera, e.g. [A, B]

        dict_X = self.scene_points
        all_X_names = npr(list(range(len(dict_X))))  # all X's, e.g. [A, B, C]
        X_values = npr(dict_X).T
        npr_X = np.nan * np.ones((4, len(dict_X)))  # TODO IF NAN SOMEWHERE IN THE NUMBERS CHECK HERE (nan)
        npr_X[:, all_X_names] = X_values

        R, t, best_inlier_Xu_idx, best_inlier_u_idx, best_inlier_indices = get_Rt_with_p3p(
            Xu_idx=X_names,
            u_idx=u_idx,
            X=npr_X,
            u_pixels=self.us[best_tentative_green_cam_id],  # should be the u's in this camera corresponding to the X's in the scene
            n_it=50,
            eps=self.threshold,
        )
        self.P2s[best_tentative_green_cam_id] = np.hstack([R, t])
        # print("done p3p ransac: ", R, t)
        # TODO THIS IS A JANKY/WRONG WAY OF DOING THIS: should choose based on a better criterium
        """
        _, unique_idx = np.unique(best_inlier_Xu_idx, return_index=True)
        X_names_to_be_kept_for_this_cam = best_inlier_Xu_idx[unique_idx]
        u_pixel_indices_to_be_kept_for_this_cam = best_inlier_u_idx[unique_idx]
        """
        X_names_to_be_kept_for_this_cam = best_inlier_Xu_idx
        u_pixel_indices_to_be_kept_for_this_cam = best_inlier_u_idx
        # print(len(self.Xus[best_tentative_green_cam_id][0]))
        for k in range(len(X_names_to_be_kept_for_this_cam)):
            self.Xus[best_tentative_green_cam_id][0].append(X_names_to_be_kept_for_this_cam[k])
            self.Xus[best_tentative_green_cam_id][1].append(u_pixel_indices_to_be_kept_for_this_cam[k])
        # print(len(self.Xus[best_tentative_green_cam_id][0]))
        # assert len(set(X_names_to_be_kept_for_this_cam)) == len(X_names_to_be_kept_for_this_cam), "scene points should(?) be unique after p3p: "+str(len(set(X_names_to_be_kept_for_this_cam)))+" vs "+str(len(X_names_to_be_kept_for_this_cam))
        self.green_cameras.remove(best_tentative_green_cam_id)
        self.red_cameras.add(best_tentative_green_cam_id)
        return best_tentative_green_cam_id

    def join_camera(self, new_cam_id):
        # go through the new camera's scene points and add them to the original cameras as 3D to 2d correspondences
        Xus = self.Xus[new_cam_id]  # (2, n)
        for red_cam_id in self.red_cameras:
            if red_cam_id == new_cam_id:
                continue
            m_red, m_new = self.get_ms(red_cam_id, new_cam_id)
            k = 0
            m_idx_to_remove = []
            # print('before shape: ', m_new.shape)
            for i in range(len(Xus[0])):
                corresp_idxs = np.where(Xus[1][i] == m_new)[0]
                if len(corresp_idxs) > 0:
                    tentative_name = Xus[0][i]
                    for corr_idx in corresp_idxs:
                        self.Xus_tentative_sps_idx[red_cam_id].append(tentative_name)
                        self.Xus_tentative_u_idx[red_cam_id].append(m_red[corr_idx])
                        m_idx_to_remove.append(corr_idx)                              # TODO: verify this
            m_red = np.delete(m_red, m_idx_to_remove)
            m_new = np.delete(m_new, m_idx_to_remove)
            # print('after shape: ', m_new.dtype)
            # for u_idx, X_idx in zip(Xus[1], Xus[0]):  # X_idx to u_idx
            #     if u_idx in m_red:
            #             self.Xus_tentative_sps_idx[red_cam_id].append(X_idx)
            #             self.Xus_tentative_u_idx[red_cam_id].append(u_idx)
            #             m_to_be_removed.append(k)
            #         k += 1
            # m_new = np.delete(m_new, m_to_be_removed)
            # m_red = np.delete(m_red, m_to_be_removed)
            self.set_ms(red_cam_id, new_cam_id, m_red, m_new)

    def get_cneighbours(self, new_cam_id):
        neighbour_ids = list()
        for i in range(self.num_of_cams):
            cid = i+1
            if new_cam_id == cid:
                continue
            if cid not in self.red_cameras:
                continue
            if len(self.get_ms(new_cam_id, cid)[0]) > 0:
                neighbour_ids.append(cid)
        # print("neighbour_ids", neighbour_ids)
        return neighbour_ids

    def reconstruct(self, new_cam_id):
        ilist = self.get_cneighbours(new_cam_id)
        print("Reconstructing for cam:", new_cam_id)
        for neighbour_cam_id in ilist:
            m_neighbour, m_new_cam = self.get_ms(neighbour_cam_id, new_cam_id)
            # reconstruct_to_3d()
            P_new = self.P2s[new_cam_id]
            P_neighbour = self.P2s[neighbour_cam_id]
            P_neighbour_to_new = np.vstack([P_new, [0, 0, 0, 1]]) @ np.linalg.inv(np.vstack([P_neighbour, [0, 0, 0, 1]]))
            F = tools.calc_F(cv06.invK, P_neighbour_to_new[:3, -1], P_neighbour_to_new[:3, :3])
            u1 = self.us[neighbour_cam_id][:, m_neighbour]  # these are homogenous image points
            u2 = self.us[new_cam_id][:, m_new_cam]
            inlier_mask = eval_inliers(u1, u2, F, self.threshold)
            m_inlier_indices = np.nonzero(inlier_mask)[0]
            # print('ms check:')
            # print(m_neighbour.shape)
            # print(npr(np.nonzero(inlier_mask)[0]))
            u1i = u1[:, inlier_mask]
            u2i = u2[:, inlier_mask]
            # print("Reconstruction: cams", new_cam_id, neighbour_cam_id, " new inliers:", u1i.shape[1])
            u1i, u2i = tools.u_correct_sampson(F, u1i, u2i)
            Xu = tools.Pu2X(cv06.K @ P_neighbour, cv06.K @ P_new, u1i, u2i)
            prev_sp_len = len(self.scene_points)
            # Xu is 4 x n, so transpose it so python can iterate row by row
            self.scene_points.extend(Xu.T)  # add the newly generated points
            # print(self.scene_points[-1])
            new_sp_len = len(self.scene_points)
            new_names = list(range(prev_sp_len, new_sp_len))
            self.Xus[new_cam_id][0].extend(new_names)
            self.Xus[new_cam_id][1].extend(m_new_cam[inlier_mask])
            self.Xus[neighbour_cam_id][0].extend(new_names)
            self.Xus[neighbour_cam_id][1].extend(m_neighbour[inlier_mask])
            self.set_ms(new_cam_id, neighbour_cam_id, npr([]), npr([]))
            self.new_x(new_cam_id, neighbour_cam_id, m_inlier_indices, new_names)

    def new_x(self, new_cam_id, neighbour_cam_id, m_inlier_indices, new_names):
        # new scene to image correspondences
        for existing_cam_id in (new_cam_id, neighbour_cam_id):
            Xus = self.Xus[existing_cam_id]
            for other_cam_id in self.green_cameras:
                m_other, m_new = self.get_ms(other_cam_id, existing_cam_id)
                # print("m_other: ", len(m_other))
                m_idx_to_remove = list()
                for i in range(len(Xus[0])):
                    corresp_idxs = np.where(Xus[1][i] == m_new)[0]
                    if len(corresp_idxs) > 0:
                        tentative_name = Xus[0][i]
                        for corr_idx in corresp_idxs:
                            self.Xus_tentative_sps_idx[other_cam_id].append(tentative_name)
                            self.Xus_tentative_u_idx[other_cam_id].append(m_other[corr_idx])
                            m_idx_to_remove.append(corr_idx)
                m_other = np.delete(m_other, m_idx_to_remove)
                # print("m_other: ", len(m_other))
                m_new = np.delete(m_new, m_idx_to_remove)
                self.set_ms(existing_cam_id, other_cam_id, m_new, m_other)

    def cluster_verification(self):
        eps2 = self.threshold
        ilist = self.red_cameras
        for ic in ilist:
            # print("red cam: ", ic)
            tentative_X_idx = npr(self.Xus_tentative_sps_idx[ic])
            tentative_u_idx = npr(self.Xus_tentative_u_idx[ic])
            # print('tentative Xs', tentative_X_idx.shape)
            # print('tentative us', tentative_u_idx.shape)
            assert len(tentative_u_idx) == len(tentative_X_idx), "tentatives lengths must match"
            # verify:
            # print(tentative_X_idx)
            if len(tentative_X_idx) == 0:
                continue
            hom_sel_X = npr(self.scene_points).T[:, tentative_X_idx]
            hom_sel_u_pixels = self.us[ic][:, tentative_u_idx]
            _P = cv06.K @ self.P2s[ic]

            temp_X = self.P2s[ic] @ hom_sel_X
            indices_infront_camera = np.where(temp_X[2, :] > 0)[0]

            X_p = hom_sel_X[:, indices_infront_camera]
            tmp_Xu_idx = tentative_X_idx[indices_infront_camera]
            tmp_u_idx = tentative_u_idx[indices_infront_camera]
            subset_of_sorted_u_pixels = tools.p2e(hom_sel_u_pixels[:, indices_infront_camera])

            u_proj = tools.p2e(_P @ X_p)

            # eval error
            err_elements = np.sum((u_proj - subset_of_sorted_u_pixels) ** 2, axis=0)
            # print(err_elements)
            maybe_inlier_indices = (err_elements < eps2)
            errs = 1 - err_elements / eps2
            errs[errs < 0] = 0
            # print(np.sum(errs))

            inlier_Xu_idx = tmp_Xu_idx[maybe_inlier_indices]
            inlier_u_idx = tmp_u_idx[maybe_inlier_indices]
            self.Xus[ic][0].extend(inlier_Xu_idx)  # from scene to image
            self.Xus[ic][1].extend(inlier_u_idx)
            # print("tent us", tentative_u_idx)
            # print('inl us', inlier_u_idx)
            # print('tentative Xs after', inlier_Xu_idx.shape)
            # print('tentative us after', inlier_u_idx.shape)

            # remove inliers as well as as outlier from the tentative correspondences
            self.Xus_tentative_sps_idx[ic] = list()
            self.Xus_tentative_u_idx[ic] = list()

    def finalize_camera(self):
        for red_cam in self.red_cameras:
            assert len(self.Xus_tentative_sps_idx[red_cam]) == 0, "tentative scene points must be empty "+str(red_cam)+": points: "+str(len(self.Xus_tentative_sps_idx[red_cam]))
            assert len(self.Xus_tentative_u_idx[red_cam]) == 0, "tentative image points must be empty "+str(red_cam)+": points: "+str(len(self.Xus_tentative_u_idx[red_cam]))


if __name__ == "__main__":
    cc = CameraContainer(12, threshold=5)
    init_cams = (1, 6)
    i1, i2, m12_inlier_idx, scene_point_names = cc.initialize(*init_cams)
    cc.start(i1, i2, m12_inlier_idx, scene_point_names)
    # Ps = [cc.P2s[init_cams[0]], cc.P2s[init_cams[1]]]
    # camera_array = ba.PySBA.KP_to_params(Ps, cv06.K)
    # cc.scene_points
    # pysba = ba.PySBA(cameraArray=camera_array, )
    for i in range(cc.num_of_cams-2):
        print("Number of red cams: ", len(cc.red_cameras))
        added_camera_id = cc.attach_camera()  # selects best tentative camera
        cc.join_camera(added_camera_id)
        cc.reconstruct(added_camera_id)
        cc.cluster_verification()
        cc.finalize_camera()

    allX = tools.p2e(npr(cc.scene_points).T)
    assert allX.shape[0]==3, "all Xs must be 3d non-homogeneous"
    np.save("allX", allX)
    cam_Ps = []
    for k in range(1, cc.num_of_cams+1):
        cam_Ps.append(cc.P2s[k])
    cam_Ps = npr(cam_Ps)
    np.save("Ps", npr(cam_Ps))
    import ge
    g = ge.GePly( 'out.ply' )
    g.points( allX )  #, ColorAll ) # Xall contains euclidean points (3xn matrix), ColorAll RGB colors (3xn or 3x1, optional)
    g.close()
    print("number of Xs:", allX.shape[1])
    print("number of Ps:", cam_Ps.shape[0])

    # conditioning
    new_Xs = np.delete(allX, np.where(np.linalg.norm(allX, axis=0) > 100), 1)

    print(np.ptp(new_Xs, axis=1))

    np.save("new_Xs", new_Xs)
    g = ge.GePly( 'out_new.ply' )
    g.points( new_Xs )  #, ColorAll ) # Xall contains euclidean points (3xn matrix), ColorAll RGB colors (3xn or 3x1, optional)
    g.close()




