import scipy.io
import numpy as np
from PIL.Image import Image

import rectify
import cv06


# WHATS imga & imgb? datatype? Image.load("img_name.png")?
root_dir = '/home/hartvi/zs22/TDV/scene_1/'
img_dir = root_dir + 'images/'

imga = None
im_a = Image.load(imga)
imgb = None
im_b = Image.load(imgb)
F_ab = cv06.ransac()  # ?

# For every selected image pair with indices i_a and i_b

#   - load the image im_a, im_b, compute fundamental matrix F_ab
#   - load corresponing points u_a, u_b
#   - keep only inliers w.r.t. F_ab

[H_a, H_b, im_a_r, im_b_r] = rectify.rectify( F_ab, im_a, im_b )

#   - modify corresponding points by H_a, H_b

#   seeds are rows of coordinates [ x_a, x_b, y ]
#      (corresponding point have the same y coordinate when rectified)
#   assuming u_a_r and u_b_r euclidean correspondences after rectification:
# seeds = np.vstack( ( u_a_r[0,:], u_b_r[0], ( u_a_r[1] + u_b_r[1] ) / 2 ) ).T

task_i = np.array( [ im_a_r, im_b_r, seeds ], dtype=object )
task += [ task_i ]

# now all stereo tasks are prepared, save to a matlab file

task = np.vstack( task )
scipy.io.savemat( 'stereo_in.mat', { 'task': task } )

# here run the gcs stereo in matlab, and then load the results

d = scipy.io.loadmat( 'stereo_out.mat' )
D = d['D']

# a disparity map for i-th pair is in D[i,0]
i = 0  # 1, 2, ...
Di = D[i,0]

plt.imshow( Di )