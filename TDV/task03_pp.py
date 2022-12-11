import numpy as np
import cv06


Xs = np.load('allX.npy')
Ps = np.load("Ps.npy")
for i in range(Ps.shape[0]):
    print("P", i+1, ":", Ps[i])
print(Xs.shape)
print(np.ptp(Xs, axis=1))

# conditioning
new_Xs = np.delete(Xs, np.where(np.linalg.norm(Xs, axis=0) > 100), 1)

print(np.ptp(new_Xs, axis=1))

np.save("new_Xs", new_Xs)

import ge
g = ge.GePly( 'out_new.ply' )
g.points( new_Xs )  #, ColorAll ) # Xall contains euclidean points (3xn matrix), ColorAll RGB colors (3xn or 3x1, optional)
g.close()

# %% PLOTTING

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

cam_positions = np.empty((3,0))

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

for k, P in enumerate(Ps):
    # P = Ps[cam_id]
    print("P", k+1, "det:", np.linalg.det(P[:3, :3]))
    P_no_K = cv06.invK @ P
    t = P_no_K[:,3:]
    R = P_no_K[:3,:3]
    C = -np.linalg.inv(R) @ t
    z = np.array([0,0,1]).reshape(-1,1)
    z_rot = R.T @ z
    cam_positions = np.concatenate((cam_positions,C),axis = 1)

    line = np.hstack([C,(C + z_rot)])
    # print(line.shape)
    ax.plot3D(line[0,:],line[1,:],line[2,:],)
    ax.text(C[0][0],C[1][0],C[2][0],k+1)

xs = cam_positions[0,:]
ys = cam_positions[1,:]
zs = cam_positions[2,:]

ax.scatter(xs,ys,zs)
ax.set_box_aspect((np.ptp(xs), np.ptp(ys), np.ptp(zs)))  # aspect ratio is 1:1:1 in data space
plt.show()




