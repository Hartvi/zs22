import numpy as np
import os
import PIL.Image
import p3p.python.p3p as p3p
import math
import tools
import cv06

npr = np.array
# p3p.p3p_grunert()


def sort_ids(i1, i2) -> tuple:
    return tuple(sorted((i1, i2)))


def get_Rt_with_p3p(Xu_idx, u_idx, X, u_pixels, n_it, eps):
    """
    Args
    ----
    - map_Xu: [n x 2] map of X indices (left column) to u indices (right column)

    """
    assert X.shape[0] == 3 or X.shape[0] == 4, "3d point have to be either 3d or 4d (hom) along columns"
    assert Xu_idx.shape == u_idx.shape, "X to u mapping have to have the same size"
    assert u_pixels.shape[0] == 2 or u_pixels.shape[0] == 3, "pixel points have to be either 2d or 3d (hom)"

    # homogenize the 3d X and 2d u_pixels
    if X.shape[0] == 3:
        X = tools.e2p(X)
    if u_pixels.shape[0] == 2:
        u_pixels = tools.e2p(u_pixels)

    eps2 = eps**2
    number_of_matches = len(Xu_idx)
    # up points need to be K undone first
    up_K_undone = tools.e2p(tools.p2e(cv06.invK @ tools.e2p(u_pixels)))
    rng = np.random.default_rng()

    best_inlier_Xu_idx, best_inlier_u_idx = None, None
    best_n_inliers = 0
    best_support = -float('inf')
    best_inlier_indices = None
    R = None
    t = None

    hom_sel_X = X[:, Xu_idx]
    hom_sel_u_pixels = u_pixels[:, u_idx]

    # iteration of RANSAC
    for it in range(n_it):
        maybe_inlier_indices = None
        three_indices = rng.choice(number_of_matches, 3, replace=False)
        three_X_idx = Xu_idx[:, three_indices]
        three_u_idx = u_idx[:, three_indices]

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
            subset_of_sorted_u_pixels = hom_sel_u_pixels[:, indices_infront_camera]

            u_proj = tools.p2e(_P @ X_p)

            # eval error
            errs = np.sum((u_proj - subset_of_sorted_u_pixels)**2, axis=0)
            errs = 1 - errs / eps2
            errs[errs < 0] = 0
            support = np.sum(errs)

            maybe_inlier_indices = (errs < eps2)

            inlier_Xu_idx = tmp_Xu_idx[maybe_inlier_indices]
            inlier_u_idx = tmp_u_idx[maybe_inlier_indices]

            if support > best_support:  # TODO is len ok here?
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
    def __init__(self, num_of_cams, root_path="C:/Users/jhart/PycharmProjects/zs22/TDV/scene_1"):
        self.num_of_cams = min(num_of_cams, 12)
        # NEW VERSION
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
        self.Xus_tentative_sps_idx = dict()  # cam_id => point name => u index
        self.Xus_tentative_u_idx = dict()  # cam_id => point name => u index
        # cam id => point name => 3d data
        self.scene_points = dict()

        for i in range(num_of_cams):
            cam_id = i+1
            points = np.loadtxt(os.path.join(root_path, "corresp", "u_{:02d}.txt".format(cam_id)))
            u = tools.e2p(points.T)
            img = os.path.join(root_path, "images", "{:02d}.jpg".format(cam_id))
            # NEW VERSION
            self.us[cam_id] = u  # cam id => u pixels in K domain
            self.ms[cam_id] = dict()
            self.Fs[cam_id] = dict()
            self.P2s[cam_id] = dict()
            self.m_inlier_indices[cam_id] = dict()
            self.Xus[cam_id] = dict()  # [3d point name] = u_index
            self.Xus_tentative_sps_idx[cam_id] = list()  # list[3d point name]
            self.Xus_tentative_u_idx[cam_id] = list()  # list[u_index]
            self.all_cameras.add(cam_id)
            self.scene_points[cam_id] = dict()  # 3d point name => 3d point
        for i in range(num_of_cams):
            for k in range(i+1, num_of_cams):
                cam1_id = i+1
                cam2_id = k+1
                corresp = np.loadtxt(os.path.join(root_path, "corresp", "m_{:02d}_{:02d}.txt".format(cam1_id, cam2_id)), dtype=int).T
                self.ms[cam1_id][cam2_id] = corresp

    def initialize(self, start_cam1, start_cam2):
        pass
        start_cam1, start_cam2 = sort_ids(start_cam1, start_cam2)
        m1, m2 = self.ms[start_cam1][start_cam2]
        u1all = self.us[start_cam1]
        u2all = self.us[start_cam2]
        u1 = u1all[:, m1]
        u2 = u2all[:, m2]
        F, P2, m_inlier_indices, Xui = cv06.ransac(u1, u2, threshold=3)
        assert m_inlier_indices.shape[0] == Xui.shape[1], "Num of 3d points Xui should equal number of inliers. Xui: "+str(Xui.shape)+" m_inlier_indices: "+str(m_inlier_indices.shape)
        self.Fs[start_cam1][start_cam2] = F
        self.P2s[start_cam1][start_cam2] = P2
        self.m_inlier_indices[start_cam1][start_cam2] = m_inlier_indices

        u1_inlier_indices = m1[m_inlier_indices]
        u2_inlier_indices = m2[m_inlier_indices]
        for i in range(m_inlier_indices.shape[0]):
            # TODO: cross check 3D point whether its new???
            Xu_name = i
            self.Xus[start_cam1][Xu_name] = u1_inlier_indices[Xu_name]
            self.Xus[start_cam2][Xu_name] = u2_inlier_indices[Xu_name]
            self.scene_points[start_cam2][Xu_name] = Xui[:, i]
            self.scene_points[start_cam2][Xu_name] = Xui[:, i]

        self.ms[start_cam1][start_cam2] = np.array([])
        self.red_cameras |= {start_cam1, start_cam2}
        return start_cam1, start_cam2, m_inlier_indices, list(range(m_inlier_indices.shape[0]))

    def start(self, i1, i2, m12_inlier_idx, scene_point_names):
        i12 = sort_ids(i1, i2)
        for cam_i in i12:
            for cam_k in range(1, self.num_of_cams+1):
                if cam_k == cam_i:
                    continue
                ord_cam_i, ord_cam_k = sort_ids(cam_i, cam_k)
                if ord_cam_i == i1 and ord_cam_k == i2:
                    continue
                corr_before = self.ms[ord_cam_i][ord_cam_k].shape[1]
                # print("number of initial matches between cam", ord_cam_i, "and", ord_cam_k, ": ", self.ms[ord_cam_i][ord_cam_k].shape)
                Xu_i = self.Xus[ord_cam_i]
                for sp_name in Xu_i:
                    u_index_in_ordered_cam_i = Xu_i[sp_name]
                    match_indices_between_ordered_cam_i_and_k = self.ms[ord_cam_i][ord_cam_k]
                    match_indices_in_ordered_cam_i, match_indices_in_ordered_cam_k = match_indices_between_ordered_cam_i_and_k
                    indices_of_match_with_ordered_cam_k = np.where(u_index_in_ordered_cam_i == match_indices_in_ordered_cam_i)[0]
                    # if len(indices_of_match_with_ordered_cam_k) > 1:
                    #     print(match_indices_in_ordered_cam_i[indices_of_match_with_ordered_cam_k])
                    for index_of_match_with_ordered_cam_k in indices_of_match_with_ordered_cam_k:
                        # ordered_cam_i already has the sp_name. Add tentative match to other camera
                        # if sp_name in self.Xus_tentative[ord_cam_k]:
                        #     print(sp_name, 'already in tentative cam', ord_cam_k)
                        self.Xus_tentative_sps_idx[ord_cam_k].append(sp_name)
                        self.Xus_tentative_u_idx[ord_cam_k].append(match_indices_in_ordered_cam_i[index_of_match_with_ordered_cam_k])
                        # delete column of the match
                        match_indices_between_ordered_cam_i_and_k = np.delete(match_indices_between_ordered_cam_i_and_k, index_of_match_with_ordered_cam_k, 1)
                        self.ms[ord_cam_i][ord_cam_k] = match_indices_between_ordered_cam_i_and_k
                # print("number of matches left between cam", ord_cam_i, "and", ord_cam_k, ": ", self.ms[ord_cam_i][ord_cam_k].shape)
                print('cam', ord_cam_k, 'tentatives:', corr_before - self.ms[ord_cam_i][ord_cam_k].shape[1])

        for tentative_cam_id in self.Xus_tentative_sps_idx:
            if len(self.Xus_tentative_sps_idx[tentative_cam_id]) > 0:
                self.green_cameras.add(tentative_cam_id)

    def attach_camera(self):
        ig = list(self.green_cameras)
        cams = np.zeros(len(self.Xus_tentative_sps_idx), dtype=int)
        tentatives = np.zeros(len(self.Xus_tentative_sps_idx), dtype=int)
        for k, cam_id in enumerate(self.Xus_tentative_sps_idx):
            cams[k] = cam_id
            tentatives[k] = len(self.Xus_tentative_u_idx[cam_id])

        best_tentative_green_cam_id = cams[np.argsort(tentatives)[-1]]
        print("best cam:", best_tentative_green_cam_id)
        u_idx = self.Xus_tentative_u_idx[best_tentative_green_cam_id]
        u_idx = npr(u_idx)
        X_names = self.Xus_tentative_sps_idx[best_tentative_green_cam_id]
        X_names = npr(X_names)
        R, t, best_inlier_Xu_idx, best_inlier_u_idx, best_inlier_indices = get_Rt_with_p3p(
            Xu_idx=X_names,
            u_idx=u_idx,
            X=self.scene_points[best_tentative_green_cam_id],
            u_pixels=self.us[best_tentative_green_cam_id],
            n_it=50,
            eps=3,
        )


if __name__ == "__main__":
    cc = CameraContainer(4)
    i1, i2, m12_inlier_idx, scene_point_names = cc.initialize(1, 2)
    cc.start(i1, i2, m12_inlier_idx, scene_point_names)
    cc.attach_camera()

    # ch = CameraHandler(cc)
    # ch.start()
    pass
    # class wrapper
    # pairwise corrs
    # tentative corresps => number of corresps that are inliers in the first cam
    #



