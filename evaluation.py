import numpy as np

def compute_iou(prediction01, gt01):
    prediction = prediction01 * 255
    prediction = prediction.astype(np.int16)
    gt = gt01 * 255
    gt = gt.astype(np.int16)
    result_and = prediction & gt
    result_or = prediction | gt
    if np.all(result_or == 0):
        iou = 1
    else:
        sum_and = np.sum(result_and) * 1.0
        sum_or = np.sum(result_or)
        iou = sum_and/sum_or

    return iou

