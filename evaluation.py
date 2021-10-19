import numpy as np

def compute_iou(prediction01, gt01):
    prediction = prediction01 #* 255
    prediction = prediction.astype(np.int16)
    gt = gt01 * 255
    gt = gt.astype(np.int16)
    result_and = prediction & gt
    result_or = prediction | gt
    if np.all(result_or == 0):
        # no foreground object in the gt, same as the prediction
        iou = 1
    else:
        sum_or = np.sum(result_or)
        if np.all(gt == 0):
            # the gt doesn't have the foreground object
            # in this case, the IOU can be what percentage background the model predicted
            iou = 1.0 - sum_or / (prediction.shape[0] * prediction.shape[1])
        else:
            # the gt includes the mask for the foreground object
            sum_and = np.sum(result_and) * 1.0
            iou = sum_and/sum_or

    return iou

