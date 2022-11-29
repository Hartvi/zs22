import numpy as np


alphabet = "abcdefghijklmnopqrstuvwxyz"

a = np.reshape(list(alphabet[:12]), (3,4))
b = np.copy(a)
width = a.shape[1]
height = a.shape[0]
print(a)


def rotate_outside(arr):
    barr = np.copy(arr)
    tmp_width = arr.shape[1]
    tmp_height = arr.shape[0]
    for i in range(1, tmp_height):
        barr[i-1, 0] = arr[i, 0]
        barr[tmp_height-i, tmp_width-1] = arr[tmp_height-1-i, tmp_width-1]

    for i in range(1, tmp_width):
        barr[0, tmp_width-i] = arr[0, tmp_width-1-i]
        barr[tmp_height-1, i-1] = arr[tmp_height-1, i]
    return barr


b_new = np.copy(a)
for i in range(1):
    for k in range(np.min((width, height)) // 2):
        # b_new[k:height-k, k:width-k] = rotate_outside(b[k:height-k, k:width-k])
        new_height = height - k - k
        new_width = width - k - k
        sub_arr_b = np.zeros((new_height, new_width), dtype=str)
        for j in range(new_height):
            for l in range(new_width):
                sub_arr_b[j, l] = b_new[k+j, k+l]
        rotated_sub_arr_b = rotate_outside(sub_arr_b)
        for j in range(new_height):
            for l in range(new_width):
                b_new[k+j, k+l] = rotated_sub_arr_b[j, l]
    b = b_new

print("a:", a)
print("b:", b)

