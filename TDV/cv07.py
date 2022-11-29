import numpy as np
import os
import PIL.Image
import p3p.python.p3p as p3p
import math
import tools
import cv06

# p3p.p3p_grunert()


def sort_ids(i1, i2) -> tuple:
    return tuple(sorted((i1, i2)))


class ScenePoint:
    scene_point_count = 0
    unique_ids = set()

    def __init__(self, space_coords, name: int):
        self.space_coords = space_coords
        assert name not in ScenePoint.unique_ids, "Trying to create point with already existing ID: " + str(name)
        self.name = name
        ScenePoint.unique_ids.add(name)
        ScenePoint.scene_point_count += 1


class SceneCamRelation:
    def __init__(self, Xu: ScenePoint, cam_id, u_id: int, is_tentative: bool):
        self.Xu = Xu
        self.cam_id = cam_id
        self.u_id = u_id
        self.is_tentative = is_tentative


class Camera:

    def __init__(self, num: int, u: np.ndarray, im: str):
        self.num = num
        assert u.shape[0] == 3, "Image coordinates must be column-wise & homogeneous! u.shape: "+str(u.shape)
        self.u = u
        self.im = PIL.Image.open(im)  # type: PIL.Image.Image
        self.scene_cam_relations = list()  # type: list[SceneCamRelation]


class CameraPair:
    def __init__(self, cam1: Camera, cam2: Camera, correspondences: np.ndarray):
        self.cam1 = cam1
        self.cam2 = cam2
        print("INIT CAMS: ", cam1.num, cam2.num)
        self.correspondences = correspondences
        self.is_empty = False
        self.F = None
        self.P2 = None
        self.m_inlier_indices = None
        self.u1_inlier_indices = None
        self.u2_inlier_indices = None

    def initiate(self):
        assert not self.is_empty, "Cannot execute pair matching twice"
        self.is_empty = True
        m1, m2 = self.correspondences
        self.correspondences = None
        u1all = self.cam1.u
        u2all = self.cam2.u
        u1 = u1all[:, m1]
        u2 = u2all[:, m2]
        F, P2, m_inlier_indices, Xui = cv06.ransac(u1, u2, threshold=3)
        assert m_inlier_indices.shape[0] == Xui.shape[1], "Num of 3d points Xui should equal number of inliers. Xui: "+str(Xui.shape)+" m_inlier_indices: "+str(m_inlier_indices.shape)
        self.F = F
        self.P2 = P2
        self.m_inlier_indices = m_inlier_indices  # type: np.ndarray
        self.u1_inlier_indices = m1[m_inlier_indices]  # index in u1, NOT a u1 value
        self.u2_inlier_indices = m2[m_inlier_indices]  # index in u2, NOT a u2 value
        # Xui[:, i] corresponds to self.cam1.u[:, self.u1_inlier_indices][:, i] and self.cam2.u[:, self.u1_inlier_indices][:, i]

        if ScenePoint.scene_point_count == 0:
            first_correspondences = True
        else:
            first_correspondences = False

        scene_points = list()  # type: list[ScenePoint]
        for i in range(m_inlier_indices.shape[0]):
            # TODO: cross check 3D point whether its new???
            if first_correspondences:
                sp = ScenePoint(space_coords=Xui[:, i], name=i)
                scene_points.append(sp)

                # indices should be in the interval [0, number of inliers]
                u1_id = self.u1_inlier_indices[i]
                u2_id = self.u2_inlier_indices[i]

                new_rel = SceneCamRelation(sp, self.cam1.num, u1_id, is_tentative=False)
                self.cam1.scene_cam_relations.append(new_rel)  # E.g. 3d point [1.1, 1.5, 2.48] & on cam

                new_rel = SceneCamRelation(sp, self.cam2.num, u2_id, is_tentative=False)
                self.cam2.scene_cam_relations.append(new_rel)
            else:
                raise NotImplementedError("Please implement checking whether a 3d point already exists")
        return m_inlier_indices, scene_points


class CameraContainer:
    def __init__(self, num_of_cams, root_path="C:/Users/jhart/PycharmProjects/zs22/TDV/scene_1"):
        self.num_of_cams = min(num_of_cams, 12)
        self.cams = []
        self.camera_pairs = dict()
        # NEW VERSION
        # pairwise
        self.ms = dict()
        self.P2s = dict()
        self.m_inlier_indices = dict()
        self.Fs = dict()
        # single
        self.us = dict()
        self.Xus = dict()  # cam_id => point name => u index
        # point name => 3d data
        self.scene_points = dict()

        for i in range(num_of_cams):
            cam_id = i+1
            points = np.loadtxt(os.path.join(root_path, "corresp", "u_{:02d}.txt".format(cam_id)))
            u = tools.e2p(points.T)
            img = os.path.join(root_path, "images", "{:02d}.jpg".format(cam_id))
            cam = Camera(cam_id, u, img)
            self.cams.append(cam)
            # NEW VERSION
            self.us[cam_id] = u
            self.ms[cam_id] = dict()
            self.Fs[cam_id] = dict()
            self.P2s[cam_id] = dict()
            self.m_inlier_indices[cam_id] = dict()
            self.Xus[cam_id] = dict()  # [3d point name] = u_index
        for i in range(num_of_cams):
            # print(i)
            for k in range(i+1, num_of_cams):
                cam1_id = i+1
                cam2_id = k+1
                corresp = np.loadtxt(os.path.join(root_path, "corresp", "m_{:02d}_{:02d}.txt".format(cam1_id, cam2_id)), dtype=int).T
                # print(corresp.shape)
                # print(self.cams[i])
                cam_pair = CameraPair(self.cams[i], self.cams[k], corresp)
                self.camera_pairs[(cam1_id, cam2_id)] = cam_pair
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
            self.scene_points[Xu_name] = Xui[:, i]

        self.ms[start_cam1][start_cam2] = np.array([])
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
                print("number of initial matches between cam", ord_cam_i, "and", ord_cam_k, ": ", self.ms[ord_cam_i][ord_cam_k].shape)
                Xu_i = self.Xus[ord_cam_i]
                for sp_name in Xu_i:
                    u_index_in_ordered_cam_i = Xu_i[sp_name]
                    match_indices_between_ordered_cam_i_and_k = self.ms[ord_cam_i][ord_cam_k]
                    match_indices_in_ordered_cam_i, match_indices_in_ordered_cam_k = match_indices_between_ordered_cam_i_and_k
                    indices_of_match_with_ordered_cam_k = np.where(u_index_in_ordered_cam_i == match_indices_in_ordered_cam_i)[0]
                    if indices_of_match_with_ordered_cam_k.shape[0] != 0:  # match exists
                        # if indices_of_match_with_ordered_cam_k.shape[0] != 1:
                        #     print(indices_of_match_with_ordered_cam_k.shape)
                        for index_of_match_with_ordered_cam_k in indices_of_match_with_ordered_cam_k:
                            # ordered_cam_i already has the sp_name:
                            self.Xus[ord_cam_k][sp_name] = match_indices_in_ordered_cam_i[index_of_match_with_ordered_cam_k]
                            # delete column of the match
                            match_indices_between_ordered_cam_i_and_k = np.delete(match_indices_between_ordered_cam_i_and_k, index_of_match_with_ordered_cam_k, 1)
                            self.ms[ord_cam_i][ord_cam_k] = match_indices_between_ordered_cam_i_and_k
                print("number of matches left between cam", ord_cam_i, "and", ord_cam_k, ": ", self.ms[ord_cam_i][ord_cam_k].shape)


class CameraHandler:
    def __init__(self, camera_container: CameraContainer):
        self.num_of_cams = camera_container.num_of_cams
        self.camera_container = camera_container

    def start(self):
        # 2.2.2  initialization
        start_cam_ids = (1, 2)
        starting_camera_pair = self.camera_container.camera_pairs[start_cam_ids]
        print("start corresp shape", starting_camera_pair.correspondences.shape)

        m_inlier_indices, scene_points = starting_camera_pair.initiate()
        print("m_inlier_indices.shape", m_inlier_indices.shape)
        print("m_inlier_indices", m_inlier_indices)
        # 2.2.3 tentative scene to image correspondences
        for start_cam_id in start_cam_ids:
            i = start_cam_id - 1  # 0, 1
            for k in range(i+1, self.num_of_cams):  # seed the cluster
                if i == 0 and k == 1:
                    continue
                cam_k_id = k + 1
                cam_pair = self.camera_container.camera_pairs[(start_cam_id, cam_k_id)]
                print("cam", start_cam_id, k, ' corresp shape: ', cam_pair.correspondences.shape)
                # m = u index => u[:, m] = points
                m_i, m_k = cam_pair.correspondences  # u1 indices & u2 indices
                # take all m1's that are matches in the seed cameras & also an unkown correspondence with camera k
                cam1_rels = cam_pair.cam1.scene_cam_relations
                cam2_rels = cam_pair.cam2.scene_cam_relations
                for j in range(len(cam1_rels)):
                    # cam1_rels[j].cam_id
                    print("m_i.shape", m_i.shape)
                    print("m_k.shape", m_k.shape)


if __name__ == "__main__":
    cc = CameraContainer(3)
    i1, i2, m12_inlier_idx, scene_point_names = cc.initialize(1, 2)
    cc.start(i1, i2, m12_inlier_idx, scene_point_names)
    # ch = CameraHandler(cc)
    # ch.start()
    pass
    # class wrapper
    # pairwise corrs
    # tentative corresps => number of corresps that are inliers in the first cam
    #



