import numpy as np

def compute_iou(prediction01, gt01):
    if np.all(gt01 == 0):
        # the gt doesn't have the foreground object
        # in this case, the IOU can be what percentage background the model predicted
        iou = 1.0 - np.count_nonzero(prediction01) / (prediction01.shape[0] * prediction01.shape[1])
    else:
        prediction = prediction01 #* 255
        prediction = prediction.astype(np.int16)
        gt = gt01 * 255
        gt = gt.astype(np.int16)

        result_and = prediction & gt
        result_or = prediction | gt

        sum_or = np.sum(result_or)
        sum_and = np.sum(result_and) * 1.0
        iou = sum_and/sum_or

    return iou

