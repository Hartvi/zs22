
import numpy as np
import matplotlib.pyplot as plt
from math import pi
from copy import copy

rng = np.random.default_rng()

def lsq(x,y):
    A = np.concatenate((np.ones((x.shape[0],1)),x.reshape(-1,1)),axis=1)
    t = np.linalg.inv(A.T @ A) @ A.T @ y.reshape(-1,1)
    return t[0],t[1]

def get_error(M,points):
    l = np.array([-M[1],1,-M[0]], dtype=np.double).reshape(1,-1)
    points = np.concatenate((points,np.ones((points.shape[0],1))),axis=1)
    points = points.T
    den = (l[0,0]**2+l[0,1]**2)**(1/2)
    error = (l @ points).T/den
    return np.fabs(error)

def evaluate_ransac(M,points,threshold):
    error = get_error(M,points)
    inliers = error < threshold
    return np.sum(inliers)

def evaluate_mlesac(M,points,threshold):
    inliers = get_inliers(M,points,threshold)
    error = get_error(M,inliers)
    mle = np.sum(1-np.square(error)/threshold**2)
    return mle

def get_inliers(M,points,threshold):
    error = get_error(M,points)
    inliers = error < threshold
    return points[inliers.ravel(),:]
    
points = np.loadtxt("linefit_3.txt")

# First LSQ on the whole set of points (non-robust; should fail):
t0,t1 = lsq(points[:,0],points[:,1])
fig,ax = plt.subplots()

ax.scatter(points[:,0],points[:,1])
ax.plot([min(points[:,0]),max(points[:,0])],[min(points[:,0])*t1+t0,max(points[:,0])*t1+t0],c='r', label='lsq')
ax.set_title("LSQ")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_xlim(min(points[:,0])-10,max(points[:,0])+10)
ax.set_ylim(min(points[:,1])-10,max(points[:,1])+10)

# -------------------------------------------------------
# -------------------------RANSAC------------------------
# -------------------------------------------------------
k = 0
N_max = 1000
n = 3
threshold = 10

best_support_RAN = 0
best_M_RAN = [0,0]

best_support_MLE = 0
best_M_MLE = [0,0]

while not k >= N_max:
    k += 1
    inds = rng.permutation(points.shape[0])[:n]
    t0,t1 = lsq(points[inds,0],points[inds,1])

    inliers_count = evaluate_ransac([t0,t1],points,threshold)
    if inliers_count > best_support_RAN:
        best_support_RAN = inliers_count
        best_M_RAN = [t0,t1]
        print(best_support_RAN,best_M_RAN)

    mle = evaluate_mlesac([t0,t1],points,threshold)
    if mle > best_support_MLE:
        best_support_MLE = mle
        best_M_MLE = [t0,t1]

x1 = min(points[:,0])
x2 = max(points[:,0])

y1 = min(points[:,0])*best_M_RAN[1]+best_M_RAN[0]
y2 = max(points[:,0])*best_M_RAN[1]+best_M_RAN[0]

RANSAC_PLOT_POINTS = np.array([[x1,x2],[y1,y2]])
print(RANSAC_PLOT_POINTS)

y1 = min(points[:,0])*best_M_MLE[1]+best_M_MLE[0]
y2 = max(points[:,0])*best_M_MLE[1]+best_M_MLE[0]

MLESAC_PLOT_POINTS = np.array([[x1,x2],[y1,y2]])

# RANSAC WITH LSQ
inliers = get_inliers(best_M_RAN,points,threshold)
t0,t1 = lsq(inliers[:,0],inliers[:,1])

RANSAC_LSQ_PLOT_POINTS = np.array([[min(points[:,0]),max(points[:,0])],[min(points[:,0])*t1+t0,max(points[:,0])*t1+t0]], dtype=np.double)

# -------------------------------------------------------
# -------------------------MLESAC------------------------
# -------------------------------------------------------
""" k = 0
best_support = 0
best_M = [0,0]

while not k >= N_max:
    k += 1
    inds = rng.permutation(points.shape[0])[:n]
    t0,t1 = lsq(points[inds,0],points[inds,1])
    mle = evaluate_mlesac([t0,t1],points,threshold)
    if mle > best_support:
        best_support = mle
        best_M = [t0,t1]

x1 = min(points[:,0])
x2 = max(points[:,0])

y1 = min(points[:,0])*best_M[1]+best_M[0]
y2 = max(points[:,0])*best_M[1]+best_M[0]

MLESAC_PLOT_POINTS = np.array([[x1,x2],[y1,y2]]) """

# MLESAC WITH LSQ
inliers = get_inliers(best_M_MLE,points,threshold)
t0,t1 = lsq(inliers[:,0],inliers[:,1])

MLESAC_LSQ_PLOT_POINTS = np.array([[min(points[:,0]),max(points[:,0])],[min(points[:,0])*t1+t0,max(points[:,0])*t1+t0]], dtype=np.double)

# ---------------------------------------------------------
# -------------------------PLOTTING------------------------
# ---------------------------------------------------------

# Plot original line
def original_line(x):
    return 10/3*x-400

ax.plot([100,400],
        [original_line(100),original_line(400)],c='cyan',zorder=8,linewidth=3, label='original')

# Plot RANSAC
print(RANSAC_PLOT_POINTS)
ax.plot(RANSAC_PLOT_POINTS[0,:],
        RANSAC_PLOT_POINTS[1,:],c='magenta',zorder=10,linewidth=2.7, linestyle='--', label='RANSAC')
        
ax.plot(RANSAC_LSQ_PLOT_POINTS[0,:],
        RANSAC_LSQ_PLOT_POINTS[1,:],c='orange',zorder=12,linewidth=2.5, linestyle='--', label='RANSAC with lsq')

# Plot MLESAC
ax.plot(MLESAC_PLOT_POINTS[0,:],
        MLESAC_PLOT_POINTS[1,:],c='green',zorder=14,linewidth=2., linestyle='--', label='MLESAC')
        
ax.plot(MLESAC_LSQ_PLOT_POINTS[0,:],
        MLESAC_LSQ_PLOT_POINTS[1,:],c='black',zorder=16,linewidth=2., linestyle='--', label='MLESAC with lsq')

ax.legend()

plt.show()


""" if min(points[:,0])*best_M[1]+best_M[0] < min(points[:,1]):
    y1 = min(points[:,1])
    x1 = (y1-best_M[0])/best_M[1]
else:
    y1 = min(points[:,0])*best_M[1]+best_M[0]
    x1 = min(points[:,0])

if max(points[:,0])*best_M[1]+best_M[0] < max(points[:,1]):
    y2 = max(points[:,1])
    x2 = (y2-best_M[0])/best_M[1]
else:
    y2 = max(points[:,0])*best_M[1]+best_M[0]
    x2 = max(points[:,0]) """