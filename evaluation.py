import numpy as np
#np.set_printoptions(threshold=np.inf)

def compute_iou(prediction01, gt01):
    prediction = prediction01 # * 255
    prediction = prediction.astype(np.int16)
    gt = gt01 * 255
    gt = gt.astype(np.int16)
    result_and = prediction & gt
    result_or = prediction | gt
    col_1 = np.count_nonzero(prediction, axis=0)
    row_1 = np.count_nonzero(prediction, axis=1)
    colgt_1 = np.count_nonzero(gt, axis=0)
    rowgt_1 = np.count_nonzero(gt, axis=1)
    colr_1 = np.count_nonzero(result_and, axis=0)
    rowr_1 = np.count_nonzero(result_and, axis=1)
    #print("pred: ",col_1, row_1,"gt:", colgt_1, rowgt_1," result:", colr_1, rowr_1)
    if np.all(result_or == 0):
        iou = 1
    else:
        sum_and = np.sum(result_and) * 1.0
        sum_or = np.sum(result_or)
        iou = sum_and/sum_or
    print(" iou: ",iou)
    return iou

