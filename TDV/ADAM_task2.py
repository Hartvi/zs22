import p5
from random import sample
import numpy as np
from tools import *
from scipy.optimize import fmin
import matplotlib.pyplot as plt

K = np.loadtxt("/mnt/e/PythonCodes/codes/TDV/p5/python/scene_1/K.txt")

def p5_wrapper(u1p, u2p, talkative = True):
    Es = p5.p5gb( u1p, u2p )    # shape of u is (3,5)

    if talkative:
        print( 'det(E)                  max alg err' )

        for E in Es:
            alg_err = np.sum( u2p * ( E @ u1p ), axis=0 )
            print( '{0}  {1} '.format( np.linalg.det( E ), alg_err.max() ) )

    return(Es)

def load_data(img_num_1, img_num_2):
    points_1 = np.loadtxt("/mnt/e/PythonCodes/codes/TDV/p5/python/scene_1/corresp/u_{}.txt".format(img_num_1)).T
    points_2 = np.loadtxt("/mnt/e/PythonCodes/codes/TDV/p5/python/scene_1/corresp/u_{}.txt".format(img_num_2)).T

    correspondences = np.loadtxt("/mnt/e/PythonCodes/codes/TDV/p5/python/scene_1/corresp/m_{}_{}.txt".format(img_num_1, img_num_2), dtype=int)

    return points_1,points_2,correspondences

def E2F(E,K):
    return np.linalg.inv(K).T @ E @ np.linalg.inv(K)


def get_inliers(F,u1,u2,threshold):
    error = err_F_sampson(F,u1,u2)
    error = error**(1/2)
    inliers = error < threshold
    return u1[:,inliers.ravel()],u2[:,inliers.ravel()]

def get_inliers_correspondences(F,u1,u2,correspondences,threshold):
    u1 = u1[:,correspondences[:,0]]
    u2 = u2[:,correspondences[:,1]]
    error = err_F_sampson(F,u1,u2)
    error = error**(1/2)
    inliers = error < threshold
    outliers = np.logical_not(inliers)
    return correspondences[inliers,:], correspondences[outliers,:]

def evaluate_mlesac(F,u1,u2,threshold):
    u1_in, u2_in = get_inliers(F,u1,u2,threshold)
    error = err_F_sampson(F,u1_in,u2_in)
    mle = np.sum(1-error/threshold**2)
    return mle,u1_in, u2_in

def ransac(points_1,points_2,correspondences,kMax=10,threshold=5):

    best_score = -np.Inf
    best_R = None
    best_t = None
    best_E = None
    best_F = None

    points_1_p = e2p(points_1[:,correspondences[:,0]])
    points_2_p = e2p(points_2[:,correspondences[:,1]])

    for i in range(kMax):
        random_correspondences = correspondences[sample(range(0,len(correspondences)),5),:]

        u1 = points_1[:,random_correspondences[:,0]]
        u2 = points_2[:,random_correspondences[:,1]]

        u1p = np.vstack( ( u1, np.ones( ( 1, 5 ) ) ) )
        u2p = np.vstack( ( u2, np.ones( ( 1, 5 ) ) ) )

        u1p_K = np.linalg.inv(K) @ u1p
        u2p_K = (u2p.T @ np.linalg.inv(K.T)).T

        Es = p5_wrapper(u1p_K, u2p_K, talkative = False)

        for E in Es:
            R,t = EutoRt(E,u1p,u2p)

            # If found solution
            if not R is None or t is None:
                F = E2F(E,K)
                mle, u1p_in, u2p_in = evaluate_mlesac(F,points_1_p,points_2_p,threshold)

                if mle > best_score:
                    P1 = np.hstack((np.eye(3),np.zeros((3,1))))
                    P2 = np.hstack((R,t))
                    X = Pu2X( P1, P2, u1p_in, u2p_in ) 
                    X2 = P2 @ X

                    in_front_mask = np.logical_and(X[2] > 0, X2[2] > 0)

                    mle, u1p_in, u2p_in = evaluate_mlesac(F,u1p_in[:,in_front_mask],u2p_in[:,in_front_mask],threshold)
                    
                    if mle > best_score:
                        best_score = mle
                        best_R = R
                        best_t = t
                        best_E = E
                        best_F = F
                        print(mle,R,t)
                        print(np.sum(in_front_mask))

    u1p_in, u2p_in = get_inliers(best_F,points_1_p,points_2_p,threshold)
    print("ransac finished")
    return(best_R,best_t,best_E,best_F,u1p_in, u2p_in)

def rodrigues(alpha,ax):
    #if ax.shape[0] == 2:
    #    ax = np.vstack((ax,1))
    ax /= vlen(ax)
    KK = sqc(ax)
    R_espanol = np.eye(3) + np.sin(alpha) * KK + (1-np.cos(alpha)) * KK @ KK
    return R_espanol
    
def optimize_E(R, t, E, u1p, u2p):
    t0 = t.reshape((-1,1))
    r0 = np.zeros((4,1))
    r0[1] = 0.01
    r0[2] = 0.01
    r0[3] = 0.01
    x0 = np.vstack((t0,r0))
        
    def optimize_me(x0):
        t = x0[:3]
        r = x0[3:]
        R0 = rodrigues(r[0],r[1:])
        E = -sqc(t) @ (R0 @ R)
        F = E2F(E,K)
        err = err_F_sampson(F, u1p, u2p)
        return np.sum(err)
    
    opt = fmin(optimize_me,x0)
    t_opt = opt[:3]
    R_opt = rodrigues(opt[3],opt[4:]) @ R
    E_opt = -sqc(t_opt) @ R_opt
    F_opt = E2F(E,K)
    return(R_opt,t_opt,E_opt,F_opt)



points_1,points_2,correspondences = load_data("01","02")    # first has to be lower than second

best_R, best_t, best_E, best_F, u1p_in, u2p_in = ransac(points_1,points_2,correspondences)
#R_opt, t_opt, E_opt, F_opt = optimize_E(best_R, best_t, best_E, u1p_in, u2p_in)

correspondences_good, correspondences_bad = get_inliers_correspondences(best_F,e2p(points_1),e2p(points_2),correspondences,5)

fig, ax = plt.subplots(1)

img = plt.imread("/mnt/e/PythonCodes/codes/TDV/p5/python/scene_1/images/01.jpg")
ax.imshow(img)

#ax.scatter(points_1[0,correspondences[:,0]],points_1[1,correspondences[:,0]],c='r',s=0.5) #,points_2[:,correspondences[i,1]]
#ax.scatter(points_2[0,correspondences[:,1]],points_2[1,correspondences[:,1]],c='b',s=0.3,zorder=3) #,points_2[:,correspondences[i,1]]

for i in range(len(correspondences_good)):
    ax.plot([points_1[0,correspondences_good[i,0]],points_2[0,correspondences_good[i,1]]],[points_1[1,correspondences_good[i,0]],points_2[1,correspondences_good[i,1]]],c='r',linewidth=0.3)
    #ax.scatter(points_1[0,correspondences[i,0]],points_1[1,correspondences[i,0]],c='r',linewidth=0.2) #,points_2[:,correspondences[i,1]]
    print(i)

for i in range(len(correspondences_bad)):
    ax.plot([points_1[0,correspondences_bad[i,0]],points_2[0,correspondences_bad[i,1]]],[points_1[1,correspondences_bad[i,0]],points_2[1,correspondences_bad[i,1]]],c='k',linewidth=0.3)
    #ax.scatter(points_1[0,correspondences[i,0]],points_1[1,correspondences[i,0]],c='r',linewidth=0.2) #,points_2[:,correspondences[i,1]]
    print(i)

plt.savefig("task2.png")