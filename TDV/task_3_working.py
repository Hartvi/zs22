# %%
import corresp
from cv06 import *
from tools import *
from random import sample
import p3p
import os.path
from copy import copy
import numpy as np
from collections import defaultdict
from scipy.optimize import fmin


# K = cv06.K
K_inv = np.linalg.inv(K)
THRESHOLD = 1

def get_inliers(F,u1,u2,threshold):
    error = err_F_sampson(F,u1,u2)
    error = error**(1/2)
    inliers = error < threshold
    return u1[:,inliers.ravel()],u2[:,inliers.ravel()]

def get_inliers_corresp_inds(F,u1,u2,correspondences,threshold):
    u1 = u1[:,correspondences[:,0]]
    u2 = u2[:,correspondences[:,1]]
    error = err_F_sampson(F,u1,u2)
    error = error**(1/2)
    inliers = error < threshold
    outliers = np.logical_not(inliers)
    return np.argwhere(inliers), np.argwhere(outliers)

def get_inliers_inds(F,u1,u2,threshold):
    error = err_F_sampson(F,u1,u2)
    error = error**(1/2)
    inliers = error < threshold
    return np.argwhere(inliers).reshape(-1)

def PP2E(P1,P2):
    P1_temp = K_inv @ P1
    R1 = P1_temp[:3,:3]
    t1 = P1_temp[:,3:]

    P2_temp = K_inv @ P2
    R2 = P2_temp[:3,:3]
    t2 = P2_temp[:,3:]

    R21 = R2 @ R1.T
    t21 = t2 - R21 @ t1

    return Rt2E(R21,t21)


def E2F(E,K):
    return np.linalg.inv(K).T @ E @ np.linalg.inv(K)

def Rt2E(R,t):
    return -sqc(t) @ R

def load_data(img_num_1, img_num_2):
    points_1 = np.loadtxt("/home/hartvi/zs22/TDV/scene_1/corresp/u_{}.txt".format(img_num_1)).T
    points_2 = np.loadtxt("/home/hartvi/zs22/TDV/scene_1/corresp/u_{}.txt".format(img_num_2)).T

    correspondences = np.loadtxt("/home/hartvi/zs22/TDV/scene_1/corresp/m_{}_{}.txt".format(img_num_1, img_num_2), dtype=int)

    return points_1,points_2,correspondences

def rodrigues(ax):
    #if ax.shape[0] == 2:
    #    ax = np.vstack((ax,1))
    if np.all(ax == np.zeros((3,1))):
        return np.eye(3)
    alpha = vlen(ax)
    ax /= vlen(ax)
    KK = sqc(ax)
    R_espanol = np.eye(3) + np.sin(alpha) * KK + (1-np.cos(alpha)) * KK @ KK
    return R_espanol
    
def optimize_Rt(R, t, image_points, scene_points):
    t0 = t.reshape((-1,1))
    r0 = np.zeros((3,1))
    x0 = np.vstack((t0,r0))
        
    def optimize_me(x0):
        t = x0[:3]
        r = x0[3:]
        R0 = rodrigues(r)
        P = K @ np.concatenate((R, t.reshape(-1,1)),axis=1)
        err = reproj_err(P,image_points,scene_points)
        return np.sum(err)
    
    opt = fmin(optimize_me,x0)
    t_opt = opt[:3]
    R_opt = R @ rodrigues(opt[3:])
    return np.asarray(R_opt,dtype=np.float64),t_opt.reshape(-1,1)

def reproj_err(P,image_points,scene_points):
    reprojected_image_points = P @ scene_points    
    err = p2e(reprojected_image_points) - p2e(image_points)
    err = vlen(err)**2
    return err

def ransac_p3p(scene_point_ids, image_point_ids, scene_points, image_points, camera_id, kMax=4000, threshold=THRESHOLD):
    best_support = -9999999999
    best_R = None
    best_t = None
    best_inliers_inds = []

    p = 0.985

    in_scene_points = scene_points[:,scene_point_ids]
    in_image_points = image_points[:,image_point_ids]

    i = 0
    while i < kMax and i < 10000:
        p3p_inds = sample(range(0,len(scene_point_ids)),3)
        p3p_scene_point_ids = scene_point_ids[p3p_inds]
        p3p_image_point_ids = image_point_ids[p3p_inds]

        p3p_scene_points = scene_points[:,p3p_scene_point_ids]
        p3p_image_points = K_inv @ image_points[:,p3p_image_point_ids]
        
        p3p_scene_points_cameras = p3p.p3p_grunert(p3p_scene_points, p3p_image_points) # TRANSPOSE 2ND ARG HERE???

        for p3p_scene_points_cam in p3p_scene_points_cameras:
            i += 1
            R,t = p3p.XX2Rt_simple(p3p_scene_points,p3p_scene_points_cam)

            Rt = np.concatenate((R,t),axis=1)
            P = K @ Rt
            
            in_front_mask = (Rt @ in_scene_points)[2,:] > 0

            in_front_image_points = in_image_points[:,in_front_mask]
            in_front_scene_points = in_scene_points[:,in_front_mask]

            err = reproj_err(P,in_front_image_points,in_front_scene_points)

            inlier_inds = err < threshold
            inliers = in_front_image_points[:,inlier_inds]

            mle = np.sum(1-err[inlier_inds]/threshold**2)
            #print(R,t)
            #print("iter: {}/{}".format(i,kMax))
            if mle > best_support:
                best_R = R
                best_t = t
                best_support = copy(mle)

                err = p2e(P @ in_scene_points) - p2e(in_image_points)
                err = vlen(err)**2
                inlier_inds = np.argwhere(np.logical_and(err < threshold, (Rt @ in_scene_points)[2,:] > 0)) # chirality here?
                print("{}/{} in front".format(np.sum(np.logical_and(err < threshold, (Rt @ in_scene_points)[2,:] > 0)),np.sum(err < threshold)))

                best_inliers_inds = inlier_inds

                eps = 1-inliers.shape[1]/in_scene_points.shape[1]
                kMax = np.log(1-p)/np.log(1-(1-eps)**3)

                print("New best support {}.".format(mle))
                print("Num inliers = {}.".format(inliers.shape[1]))
                #print("New best P = {}.".format(P))

    return best_R,best_t,best_inliers_inds.reshape(-1)

corresp_handler = corresp.Corresp(12)
corresp_handler.verbose = 2

all_points_dict = dict()
all_correspondences_dict = dict()

most_corresps_pair = [0,0,0]

for i in range(11):
    ind_1 = i+1

    if ind_1<=9:
        ind_1_str = "0" + str(ind_1)
    else:
        ind_1_str = str(ind_1)

    for j in range(ind_1,12):    
        ind_2 = j+1

        if ind_2<=9:
            ind_2_str = "0" + str(ind_2)
        else:
            ind_2_str = str(ind_2)

        points_1, points_2, correspondences = load_data(ind_1_str, ind_2_str)

        if not ind_1-1 in all_points_dict:
            all_points_dict[ind_1-1] = e2p(points_1)

        if not ind_2-1 in all_points_dict:
            all_points_dict[ind_2-1] = e2p(points_2)

        if not ind_1-1 in all_correspondences_dict:
            all_correspondences_dict[ind_1-1] = dict()

        all_correspondences_dict[ind_1-1][ind_2-1] = correspondences

        if correspondences.size > most_corresps_pair[2]:
            most_corresps_pair = [ind_1-1,ind_2-1,correspondences.size]

        corresp_handler.add_pair(ind_1-1, ind_2-1, np.array(correspondences))

P0 = K @ np.hstack((np.eye(3),np.zeros((3,1))))

init_id_0 = 0#most_corresps_pair[0]
init_id_1 = 1#most_corresps_pair[1]
print(most_corresps_pair)

try:
    init_cam_pair_R = np.load("/mnt/e/PythonCodes/codes/TDV/init_cam_pair_R.npy")
    init_cam_pair_t = np.load("/mnt/e/PythonCodes/codes/TDV/init_cam_pair_t.npy")
    init_cam_pair_E = Rt2E(init_cam_pair_R,init_cam_pair_t)
    init_cam_pair_F = E2F(init_cam_pair_E,K)
    print("Loading initial camera pair R,t from file.")
except:    
    best_R, best_t, best_E, best_F, u1p_in, u2p_in = ransac(all_points_dict[init_id_0],all_points_dict[init_id_1],all_correspondences_dict[init_id_0][init_id_1], kMax=100)
    init_cam_pair_R, init_cam_pair_t, init_cam_pair_E, init_cam_pair_F = optimize_E(best_R, best_t, best_E, u1p_in, u2p_in)
    
    init_cam_pair_R = np.asarray(init_cam_pair_R,dtype=np.float64)
    init_cam_pair_t = np.asarray(init_cam_pair_t,dtype=np.float64)
    
    np.save("init_cam_pair_R.npy",np.array(init_cam_pair_R))
    np.save("init_cam_pair_t.npy",np.array(init_cam_pair_t))
    print("Initial camera pair obtained R={} t={}".format(init_cam_pair_R,init_cam_pair_t))

P1 = K @ np.concatenate((init_cam_pair_R, init_cam_pair_t.reshape(-1,1)),axis=1)

Ps = dict()
Ps[init_id_0] = P0 
Ps[init_id_1] = P1 

correspondences_in, correspondences_out = get_inliers_corresp_inds(init_cam_pair_F,all_points_dict[init_id_0],all_points_dict[init_id_1],all_correspondences_dict[init_id_0][init_id_1],THRESHOLD)
correspondences_in = correspondences_in.reshape(-1)

in_corr_inds = all_correspondences_dict[init_id_0][init_id_1][correspondences_in,:]
scene_points = Pu2X( Ps[init_id_0], Ps[init_id_1], all_points_dict[init_id_0][:,in_corr_inds[:,0]], all_points_dict[init_id_1][:,in_corr_inds[:,1]])

corresp_handler.start(init_id_0, init_id_1, correspondences_in.reshape(-1))
exit()
# %%
# list of cameras with tentative scene-to-image correspondences
camera_ids, _ = corresp_handler.get_green_cameras()
# counts of tentative correspondences in each ‘green’ camera
Xucount,_ = corresp_handler.get_Xucount( camera_ids )

while camera_ids.size != 0:
    cam_ind = np.argmax(Xucount)
    camera_id = camera_ids[cam_ind]

    #Xucount = np.delete(Xucount,cam_ind)
    #camera_ids = np.delete(camera_ids,cam_ind)

    scene_point_ids, image_point_ids, xu_verified = corresp_handler.get_Xu( camera_id )  # get scene-to-image correspondences

    if len(scene_point_ids) >= 3:   # check enough correspondences
        image_points = all_points_dict[camera_id]
        R,t,xinl = ransac_p3p(scene_point_ids, image_point_ids, scene_points, image_points, camera_id)
        
        R,t = optimize_Rt(R, t, copy(image_points[:,image_point_ids[xinl]]), copy(scene_points[:,scene_point_ids[xinl]]))

        P_new = K @ np.concatenate((R, t.reshape(-1,1)),axis=1)
        # REFINE BY NUMERICAL MINIMISATION
        
        Ps[camera_id] = P_new 
        
        corresp_handler.join_camera( camera_id, xinl )

        ilist = corresp_handler.get_cneighbours( camera_id )    # List of cameras in the cluster that are related to the attached
                                                                # camera by some image-to-image correspondences.
        for ic in ilist:
            [ m1, m2 ] = corresp_handler.get_m( camera_id, ic )    # get remaining image-to-image correspondences

            # Reconstruct new scene points using the cameras i and ic and image-to-image correspondences m. Sets
            # of inliers and new scene points’ IDs are obtained.

            u1 = all_points_dict[camera_id][:,m1]
            u2 = all_points_dict[ic][:,m2]

            P1 = Ps[camera_id]
            P2 = Ps[ic]
            E = PP2E(P1,P2)
            F = E2F(E,K)

            m_inliers = get_inliers_inds(F,u1,u2,THRESHOLD)

            new_scene_points = Pu2X( P1, P2, u1[:,m_inliers], u2[:,m_inliers])

            scene_points = np.concatenate((scene_points,new_scene_points), axis=1)
            print("\nTOTAL SCENE POINTS {}".format(scene_points.shape[1]))
            
            corresp_handler.new_x( camera_id, ic, m_inliers, None )

        # VERIFICATION
        cams_to_verify = corresp_handler.get_selected_cameras() # list of all cameras in the cluster
        
        for cam_to_verify in cams_to_verify:

            # Verify (by reprojection error) scene-to-image correspondences in Xu_tentative. A subset of good
            # points is obtained.

            [X_ids, u_ids, Xu_verified] = corresp_handler.get_Xu( cam_to_verify )

            image_points_cam = all_points_dict[cam_to_verify][:,u_ids]
            scene_points_cam = scene_points[:,X_ids]

            Xu_tentative = np.logical_not(Xu_verified)  # There is sometimes a weirdly small number of these. like 2

            X_tentative = scene_points_cam[:,Xu_tentative]
            u_tentative = image_points_cam[:,Xu_tentative]

            P = Ps[cam_to_verify]

            u_reprojected = P @ X_tentative    

            err = p2e(u_reprojected) - p2e(u_tentative)
            err = vlen(err)**2  # either EXTREMLY LARGE or really small...
            inlier_inds = err < THRESHOLD

            corr_ok = np.argwhere(Xu_tentative)[inlier_inds]

            corresp_handler.verify_x( cam_to_verify, corr_ok )

        corresp_handler.finalize_camera()

        camera_ids, _ = corresp_handler.get_green_cameras()
        Xucount,_ = corresp_handler.get_Xucount( camera_ids )



import matplotlib.pyplot as plt

cam_positions = np.empty((3,0))

for cam_id in Ps:
    P = Ps[cam_id]
    P_no_K = K_inv @ P
    t = P_no_K[:,3:]
    R = P_no_K[:3,:3]
    C = np.linalg.inv(-R) @ t
    cam_positions = np.concatenate((cam_positions,C),axis = 1)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

xs = cam_positions[0,:]
ys = cam_positions[1,:]
zs = cam_positions[2,:]

ax.scatter(xs,ys,zs)
ax.set_box_aspect((np.ptp(xs), np.ptp(ys), np.ptp(zs)))  # aspect ratio is 1:1:1 in data space
plt.show()

# %%
p2e(scene_points)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

xs = scene_points[0,:]
ys = scene_points[1,:]
zs = scene_points[2,:]

ax.scatter(xs,ys,zs)
ax.set_box_aspect((np.ptp(xs), np.ptp(ys), np.ptp(zs)))  # aspect ratio is 1:1:1 in data space
plt.show()

# %%
import ge
g = ge.GePly( 'out_opt_7-11.ply' )
g.points( scene_points[:3,:] ) # Xall contains euclidean points (3xn matrix), ColorAll RGB colors (3xn or 3x1, optional)
g.close()


