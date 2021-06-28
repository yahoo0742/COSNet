import numpy as np
import cv2
import random

def flip(I,flip_p):
    if flip_p>0.5:
        return np.fliplr(I)
    else:
        return I

def scale(img, scale, interpolation = cv2.INTER_LINEAR):
    new_dims = (int(img.shape[0]*scale),  int(img.shape[1]*scale))
    return cv2.resize(img,new_dims, interpolation).astype(float)

def crop(img, gt):
    H = int(0.9 * img.shape[0])
    W = int(0.9 * img.shape[1])
    H_offset = random.choice(range(img.shape[0] - H))
    W_offset = random.choice(range(img.shape[1] - W))
    H_slice = slice(H_offset, H_offset + H)
    W_slice = slice(W_offset, W_offset + W)
    img = img[H_slice, W_slice, :]
    gt = gt[H_slice, W_slice]

    return img, gt