import numpy as np
import cv2
import random

def flip2d(I, flip_p):
    if flip_p>0.5:
        return np.fliplr(I)
    else:
        return I

def flip3d(I, flip_p):
    result_img = []
    for img2d in I:
        new_img = flip2d(img2d, flip_p)
        result_img.append(new_img)
    return np.array(result_img)

def scale2d(img, scale, interpolation = cv2.INTER_LINEAR):
    new_height = int(img.shape[0]*scale)
    new_width = int(img.shape[1]*scale)
    new_dims = (new_width, new_height)
    # print("**** scale2d new dims:",new_dims, " original size ",img.shape)
    return cv2.resize(src=img, dsize=new_dims, interpolation=interpolation)

def scale3d(img, scale, interpolation = cv2.INTER_LINEAR):
    result_img = []
    for img2d in img:
        new_img = scale2d(img2d, scale, interpolation)
        result_img.append(new_img)
    return np.array(result_img)

def crop2d(img, size_scale, offset=None):
    H = int(size_scale * img.shape[0])
    W = int(size_scale * img.shape[1])
    if offset == None:
        H_offset = random.choice(range(img.shape[0] - H))
        W_offset = random.choice(range(img.shape[1] - W))
        offset = {'x': W_offset, 'y': H_offset}
    else:
        H_offset = offset['y']
        W_offset = offset['x']
    H_slice = slice(H_offset, H_offset + H)
    W_slice = slice(W_offset, W_offset + W)
    img = img[H_slice, W_slice]

    return img, offset

def crop3d(img, size_scale, offset=None): #-> np.array:
    # img should be in the shape of [C, H, W]
    result_img = []
    for img2d in img:
        result, offset = crop2d(img2d, size_scale, offset)
        result_img.append(result)
    result_img = np.array(result_img)
    return result_img, offset