 
import matplotlib.pyplot as plt
 
from mpl_toolkits import mplot3d
import numpy as np
import cv06
 
cam_positions = np.empty((3,0))
 
 
fig = plt.figure()
 
ax = fig.add_subplot()
 
#ax = fig.add_subplot(projection='3d')
 
 
Xs = np.load('allX.npy')
Ps = np.load("Ps.npy")
for cam_id in range(len(Ps)):
    
    P = Ps[cam_id]
    
    P_no_K = cv06.invK @ P
    
    t = P_no_K[:,3:]
    
    R = P_no_K[:3,:3]
    
    C = -np.linalg.inv(R) @ t
    
    z = np.array([0,0,1]).reshape(-1,1)
    
    z_rot = R.T @ z
    
    cam_positions = np.concatenate((cam_positions,C),axis = 1)
    
    
    line = np.hstack([C,(C + z_rot)])
    
    #ax.plot(line[0,:],line[1,:])
    
    ax.text(C[0][0],C[1][0],cam_id)
 
 
xs = cam_positions[0,:]
 
ys = cam_positions[1,:]
 
 
ax.scatter(xs,ys)
plt.show()