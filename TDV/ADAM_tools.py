import math
import numpy as np
from scipy.spatial.transform import Rotation   

def e2p(u_e):
    return np.vstack((u_e, np.ones((1,u_e.shape[1]))))

def p2e(u_p):
    u_p /= u_p[-1:,:]
    return u_p[:-1,:]

def vlen(x):
    return np.sqrt(np.sum(x**2,axis=0))

def sqc(x):
    if len(x.shape) == 2:
        return np.array([[0, -x[2,:], x[1,:]],
                        [x[2,:], 0, -x[0,:]],
                        [-x[1,:], x[0,:], 0]])
    else:
        return np.array([[0, -x[2], x[1]],
                        [x[2], 0, -x[0]],
                        [-x[1], x[0], 0]])

def EutoRt(E,u1,u2):
    
    u1 /= u1[-1,:]
    u2 /= u2[-1,:]

    [U,D,V_T] = np.linalg.svd(E)

    W1 = np.array([[0,1,0],[-1,0,0],[0,0,1]])
    W2 = np.array([[0,-1,0],[1,0,0],[0,0,1]])

    R1 = U @ W1 @ V_T
    R2 = U @ W2 @ V_T

    t1 = -U[:,-1:]
    t2 = U[:,-1:]

    P1 = np.concatenate((R1,t1),axis=1)
    P2 = np.concatenate((R1,t2),axis=1)
    P3 = np.concatenate((R2,t1),axis=1)
    P4 = np.concatenate((R2,t2),axis=1)

    P_I = np.hstack((np.eye(3),np.zeros((3,1))))
    Ps = [P1,P2,P3,P4]
    
    Ps_in_front_camera = []

    for P in Ps:
        is_in_front = True
        for i in range(u1.shape[1]):

            """ D = [P_I[0,:], - P_I[2,:] * u1[0,i],
                P_I[1,:], - P_I[2,:] * u2[0,i],
                P[0,:], - P[2,:] * u1[1,i],
                P[1,:], - P[2,:] * u2[1,i]] """

            D = np.array([u1[0,i] * P_I[2,:] - P_I[0,:],
                        u1[1,i] * P_I[2,:] - P_I[1,:],
                        u2[0,i] * P[2,:] - P[0,:],
                        u2[1,i] * P[2,:] - P[1,:]])

            S = np.diag(1/np.max(np.fabs(D), axis=0))
            D_conditioned = D @ S
        
            #U,_,_ = np.linalg.svd(D_conditioned.T @ D_conditioned)
            U,_,V_T = np.linalg.svd(D_conditioned)
            X = V_T[-1,:]
            #X = U[:,-1]
            X = S @ X       
            X /= X[-1]

            """ u1_check = P_I @ X
            u2_check = P @ X

            print(i)
            print(X)
            print(P)
            print(u1_check/u1_check[-1],u1[:,i])
            print(u2_check/u2_check[-1],u2[:,i]) """

            #_,_,V_A_T = np.linalg.svd(D)
            #X = V_A_T[-1,:]

            X2 = P @ X

            if X[2] <= 0 or X2[2] <= 0:
                is_in_front = False
                break

        if is_in_front:
            Ps_in_front_camera.append(P)

    if len(Ps_in_front_camera) == 0:
        #return np.eye(3),t1
        return np.empty((3,3)),np.empty((3,1))
    elif len(Ps_in_front_camera) == 1:
        #print("ok")
        return Ps_in_front_camera[0][:,:3],Ps_in_front_camera[0][:,-1].reshape(-1,1)
    else:
        #print(len(Ps_in_front_camera))
        print("tohle by nemelo nastat")
        #return np.eye(3),t1
        #print("ooooooof")
        return Ps_in_front_camera[0][:,:3],Ps_in_front_camera[0][:,-1].reshape(-1,1)

def Pu2X( P1, P2, u1, u2 ):

    Xs = np.zeros((4,u1.shape[1]))
    u1 /= u1[-1,:]
    u2 /= u2[-1,:]

    """ X_ref = np.array(
[[-0.2965874,  -0.20309584,  0.06980417, -0.09555654, -0.25906339,  0.08736477,
   0.0848115,   0.01569596,  0.08690224, -0.0493256 ],
 [-0.20331721, -0.24252846,  0.00977047, -0.14562052, -0.06309828,  0.05263832,
   0.0891896,  -0.17967297,  0.1632732,  -0.20101907],
 [-0.91391429, -0.91483634,  0.97961459, -0.94521391, -0.95117835,  0.97450076,
   0.95800934, -0.95521034,  0.95146989, -0.93889   ],
 [-0.1883048,  -0.25101096,  0.18811476, -0.27610554, -0.15544938,  0.19986213,
   0.2589794,  -0.23461555,  0.24595714, -0.27503435]]) """

    for i in range(u1.shape[1]):
        D = np.array([u1[0,i] * P1[2,:] - P1[0,:],
                    u1[1,i] * P1[2,:] - P1[1,:],
                    u2[0,i] * P2[2,:] - P2[0,:],
                    u2[1,i] * P2[2,:] - P2[1,:]])

        S = np.diag(1/np.max(np.fabs(D), axis=0))
        D_conditioned = D @ S
    
        U,_,_ = np.linalg.svd(D_conditioned.T @ D_conditioned)
        X = U[:,-1]
        X = S @ X

        """ print(vlen(D@X))
        print(vlen(X))
        print(vlen(D@X_ref[:,i]))
        print(vlen(X_ref[:,i]))

        print("------") """
        Xs[:,i] = X#/X[-1]#np.vstack((X,1))

    return Xs

    """ Xs = np.zeros((4,u1.shape[1]))

    for i in range(u1.shape[1]):
        D = [P1[0,:], - P1[2,:] * u1[0,i],
            P1[1,:], - P1[2,:] * u2[0,i],
            P2[0,:], - P2[2,:] * u1[1,i],
            P2[1,:], - P2[2,:] * u2[1,i]]

        _,_,V_A_T = np.linalg.svd(D)
        X = V_A_T[-1:,:]
        Xs[:,i] = np.vstack((X,1))

    return Xs """

def err_F_sampson(F,u1,u2):
    errs = np.zeros(u1.shape[1])

    """ for i in range(u1.shape[1]):
        num = (u2[:,i].T @ F @ u1[:,i])**2
        den = (((F @ u1[:,i])**2)[0] + ((F @ u1[:,i])**2)[1] + ((F.T @ u2[:,i])**2)[0] + ((F.T @ u2[:,i])**2)[1])
        err = num/den
        errs[i] = err """

    S = np.array([[1,0,0],[0,1,0]])

    u1 /= u1[-1,:]
    u2 /= u2[-1,:]
    
    for i in range(u1.shape[1]):
        num = (u2[:,i].T @ F @ u1[:,i])
        den = np.sqrt(vlen(S @ F @ u1[:,i])**2 + vlen(S @ F.T @ u2[:,i])**2)
        err = num/den
        errs[i] = err

    return errs**2
    #return errs**(1/2)

def u_correct_sampson(F,u1,u2):
    S = np.array([[1,0,0],[0,1,0]])

    u1 /= u1[-1,:]
    u2 /= u2[-1,:]

    nu1 = np.ones((u1.shape))
    nu2 = np.ones((u1.shape))

    for i in range(u1.shape[1]):
        orig_us = np.vstack((u1[0:2,i:i+1],u2[0:2,i:i+1]))
        num = (u2[:,i].T @ F @ u1[:,i])
        num2 = np.vstack((F[:,0].T @ u2[:,i], F[:,1].T @ u2[:,i], F[0,:] @ u1[:,i], F[1,:] @ u1[:,i]))
        den = vlen(S @ F @ u1[:,i])**2 + vlen(S @ F.T @ u2[:,i])**2

        err = num/den
        nu12 = err * num2
        nu12 = orig_us - nu12

        nu1[0:2,i] = nu12[:2].reshape(-1)
        nu2[0:2,i] = nu12[2:].reshape(-1)

    return nu1,nu2

def calc_F(invK, t, R):
    F = invK.T @ sqc(-t) @ R @ invK
    return F

def rodrigues(theta, axis):
    sqc_axis = sqc(axis)
    return np.eye(3) + sqc_axis * math.sin(theta) + sqc_axis @ sqc_axis * (1 - math.cos(theta))


def err_contributions(errs, theta=3):
    # print(1 - errs / theta**2)
    return np.clip(1 - errs / theta ** 2, 0, 1)

""" E = np.array(
[[ 0.06284,    -0.4346839,   0.39247907],
 [ 0.17839508,  0.23155639, -0.81206822],
 [-0.46489848,  0.77322974,  0.30683618]])
u1 = np.array(
[[0.34656371, 1.51341325, 1.97559411, 0.62821997, 1.16674273],
 [0.78104715, 0.52594082, 0.33271419, 1.08906851, 1.62547932],
 [5.028389,   5.77545078, 5.57365427, 5.66253634, 5.94839749]])
u2 = np.array(
[[0.37055458, 1.40644025, 1.8939061,  0.55093476, 1.0397616 ],
 [2.68877948, 2.70450048, 2.48177819, 3.17498675, 3.79248974],
 [4.89159804, 5.84627709, 5.77963369, 5.44604888, 5.64403445]])

print(EutoRt(E,u1,u2)) """

""" F = np.array(
[[ 5.34686657e-08, -3.04957920e-08,  3.19414599e-04],
 [ 7.93564101e-08, -9.72549933e-08, -4.72262388e-04],
 [-5.38993851e-04,  4.80728376e-04, -6.54536468e-02]])
u1 = np.array(
[[486.74333264, 669.70066858, 956.31639813, 914.88282106, 487.06049756,
  966.68644244, 744.8702797,  768.80124467, 579.35008153, 503.32052411],
 [566.76779127, 755.26189694, 420.2755637,  897.93970338, 479.88766778,
  679.63450944, 513.18840764, 721.30632058, 739.8086245,  636.28386849],
 [  1.,           1.,           1.,           1.,           1.,
    1.,           1.,           1.,           1.,           1.,        ]])
u2 = np.array(
[[ 5.60966686e+02,  7.86827045e+02,  1.02707984e+03,  1.02247874e+03,
   5.17860027e+02,  9.53502406e+02,  7.43827173e+02,  9.52696118e+02,
   5.79490304e+02,  6.17033081e+02],
 [ 2.67108500e+02,  4.18706690e+02, -2.85187674e+01,  4.60916955e+02,
   1.61006638e+02,  2.21033431e+02,  1.20297312e+02,  3.86101663e+02,
   3.44141184e+02,  3.55870137e+02],
 [ 1.00000000e+00,  1.00000000e+00,  1.00000000e+00,  1.00000000e+00,
   1.00000000e+00,  1.00000000e+00,  1.00000000e+00,  1.00000000e+00,
   1.00000000e+00,  1.00000000e+00]]
)
print(u_correct_sampson(F,u1,u2)) """

""" P1 = np.array([[1.5e+03, 0.0e+00, 5.0e+02, 0.0e+00],
 [0.0e+00, 1.5e+03, 4.0e+02, 0.0e+00],
 [0.0e+00, 0.0e+00, 1.0e+00, 0.0e+00]])
P2 = np.array([[ 1.74258601e+03,  4.56561223e+02, -7.03267921e+01,  1.55563492e+03],
 [-1.62058905e+02,  1.43401648e+03,  1.02826730e+03,  1.01015254e+03],
 [ 4.38525402e-01, -2.25572041e-01,  8.69949841e-01,  3.03045763e-01]])
u1 = np.array([[5.50520897e+03, 5.02876830e+03, 3.00524155e+03, 4.16127521e+03,
  4.69678561e+03, 3.88998225e+03, 4.05022839e+03, 2.78966569e+03,
  3.80878629e+03, 4.09200954e+03],
 [4.09327646e+03, 4.81539689e+03, 2.05485362e+03, 4.03004225e+03,
  2.58224272e+03, 2.94916418e+03, 3.45405295e+03, 4.00327006e+03,
  3.93075932e+03, 5.09839603e+03],
 [5.87767028e+00, 6.02038520e+00, 5.39685040e+00, 6.31815945e+00,
  5.86254285e+00, 6.28965240e+00, 6.51535173e+00, 5.47662698e+00,
  5.98316759e+00, 6.69998647e+00]])
u2 = np.array([[4.97552777e+03, 4.65406412e+03, 1.79582056e+03, 3.39793647e+03,
  3.69925094e+03, 2.47538002e+03, 2.99652829e+03, 1.95612616e+03,
  3.03497679e+03, 3.36801127e+03],
 [8.15167977e+03, 8.88180299e+03, 6.15996691e+03, 8.23296408e+03,
  6.68278082e+03, 7.48117326e+03, 7.78189620e+03, 8.21014127e+03,
  8.10661028e+03, 9.51015592e+03],
 [6.22274174e+00, 5.87006740e+00, 5.77900287e+00, 6.15363025e+00,
  6.32080724e+00, 6.07059065e+00, 6.20904836e+00, 5.31858326e+00,
  5.03857157e+00, 6.25962336e+00]])

print(Pu2X( P1, P2, u1, u2 )) """