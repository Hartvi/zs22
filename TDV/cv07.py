import numpy as np
import os
import PIL.Image
import p3p.python.p3p as p3p
import math
import tools
import cv06

npr = np.array
# p3p.p3p_grunert()


def isin_w_duplicates(x, y):
    ret1 = np.zeros(len(x), dtype=bool)
    ret2 = np.zeros(len(y), dtype=bool)
    # return np.array([(_ in y) for _ in x])
    checksum = 0
    # for example doesnt work fo x = [1,1,2] and y = [1,2] => sumy = 2, sumx = 3
    for k, i in enumerate(x):
        ret1[k] = i in y
        checksum += 1
    for k, i in enumerate(y):
        ret2[k] = i in x
        checksum -= 1
    assert checksum == 0, "wrong isin duplicates"
    return ret1, ret2


def sort_ids(i1, i2) -> tuple:
    return tuple(sorted((i1, i2)))


def get_Rt_with_p3p(Xu_idx, u_idx, X, u_pixels, n_it, eps):
    """
    Args
    ----
    - map_Xu: [n x 2] map of X indices (left column) to u indices (right column)

    """
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
            errs = np.sum((u_proj - subset_of_sorted_u_pixels)**2, axis=0)
            maybe_inlier_indices = (errs < eps2)
            errs = 1 - errs / eps2
            errs[errs < 0] = 0
            support = np.sum(errs)

            inlier_Xu_idx = tmp_Xu_idx[maybe_inlier_indices]
            inlier_u_idx = tmp_u_idx[maybe_inlier_indices]

            if support > best_support:

                n_it = tools.Nmax(0.99, np.sum(maybe_inlier_indices) / number_of_matches, 3)
                R = _R
                t = _t
                best_inlier_indices = maybe_inlier_indices
                best_inlier_Xu_idx = inlier_Xu_idx
                best_inlier_u_idx = inlier_u_idx
                best_support = support
                print("max_iters: ", n_it, "p3p best support: ", support)

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
        # point name => 3d data
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
        for i in range(num_of_cams):
            for k in range(i+1, num_of_cams):
                cam1_id = i+1
                cam2_id = k+1
                corresp = np.loadtxt(os.path.join(root_path, "corresp", "m_{:02d}_{:02d}.txt".format(cam1_id, cam2_id)), dtype=int).T
                self.ms[cam1_id][cam2_id] = corresp
                self.ms[cam2_id][cam1_id] = corresp

    def initialize(self, start_cam1, start_cam2):
        pass
        start_cam1, start_cam2 = sort_ids(start_cam1, start_cam2)
        m1, m2 = self.ms[start_cam1][start_cam2]
        u1all = self.us[start_cam1]
        u2all = self.us[start_cam2]
        u1 = u1all[:, m1]
        u2 = u2all[:, m2]
        F, P2, m_inlier_indices, X = cv06.ransac(u1, u2, threshold=3)
        assert m_inlier_indices.shape[0] == X.shape[1], "Num of 3d points X should equal number of inliers. X: "+str(X.shape)+" m_inlier_indices: "+str(m_inlier_indices.shape)
        # print("m_inlier_indices:", m_inlier_indices)
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
            self.scene_points[Xu_name] = X[:, i]

        self.ms[start_cam1][start_cam2] = np.array([])
        self.red_cameras |= {start_cam1, start_cam2}
        return start_cam1, start_cam2, m_inlier_indices, list(range(m_inlier_indices.shape[0]))

    def start(self, i1, i2, m12_inlier_idx, scene_point_names):
        i12 = sort_ids(i1, i2)
        # ms = self.ms[i12[0]][i12[1]]  # empty because weve used them
        m12_sp_names = {i1: npr(list(self.Xus[i1].keys())), i2: npr(list(self.Xus[i2].keys()))}
        m12_inliers = {i1: npr(list(self.Xus[i1].values())), i2: npr(list(self.Xus[i2].values()))}
        for cam_i in i12:
            for cam_k in range(1, self.num_of_cams+1):
                if cam_k == cam_i:
                    continue
                if cam_i == i1 and cam_k == i2:
                    continue
                if cam_k == i1 and cam_i == i2:
                    continue
                if cam_i < cam_k:
                    cam_i_effective_idx = cam_i
                    cam_k_effective_idx = cam_k
                else:
                    cam_i_effective_idx = cam_k
                    cam_k_effective_idx = cam_i
                mi, mk = self.ms[cam_i_effective_idx][cam_k_effective_idx]

                mk_idx_to_mi, original_inliers_to_cam_k = isin_w_duplicates(m12_inliers[cam_i], mi)  # can have duplicate scene to screen correspondences
                # original_inliers_to_cam_k = isin_w_duplicates(mi, m12_inliers[cam_i])  # can have duplicate scene to screen correspondences
                # print("mk_idx_to_mi", mk_idx_to_mi.shape, np.sum(mk_idx_to_mi))
                print("original_inliers_to_cam_k", original_inliers_to_cam_k.shape, np.sum(original_inliers_to_cam_k))
                self.Xus_tentative_sps_idx[cam_k] += m12_sp_names[cam_i][mk_idx_to_mi].tolist()
                self.Xus_tentative_u_idx[cam_k] += mk[original_inliers_to_cam_k].tolist()
                assert len(self.Xus_tentative_sps_idx[cam_k]) == len(self.Xus_tentative_u_idx[cam_k]), "handle duplicates"
                print("self.Xus_tentative_sps_idx[cam_k]", len(self.Xus_tentative_sps_idx[cam_k]))
                print("self.Xus_tentative_u_idx[cam_k]", len(self.Xus_tentative_u_idx[cam_k]))
                tmp_ms = self.ms[cam_i_effective_idx][cam_k_effective_idx]
                # remove the added correspondences
                self.ms[cam_i_effective_idx][cam_k_effective_idx] = tmp_ms[:, ~original_inliers_to_cam_k]

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
        X_names = npr(list(X_names))  # X's found in the green camera, e.g. [A, B]

        dict_X = self.scene_points  # type: dict
        all_X_names = npr(list(dict_X.keys()))  # all X's, e.g. [A, B, C]
        X_values = npr(list(dict_X.values())).T
        npr_X = np.nan * np.ones((4, len(dict_X)))  # TODO IF NAN SOMEWHERE IN THE NUMBERS CHECK HERE (nan)
        npr_X[:, all_X_names] = X_values

        R, t, best_inlier_Xu_idx, best_inlier_u_idx, best_inlier_indices = get_Rt_with_p3p(
            Xu_idx=X_names,
            u_idx=u_idx,
            X=npr_X,
            u_pixels=self.us[best_tentative_green_cam_id],  # should be the u's in this camera corresponding to the X's in the scene
            n_it=50,
            eps=3,
        )
        print(R, t)


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



