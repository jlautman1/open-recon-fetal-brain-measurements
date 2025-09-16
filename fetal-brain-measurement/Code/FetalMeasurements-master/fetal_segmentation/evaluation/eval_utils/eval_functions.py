import numpy as np
from fetal_segmentation.evaluation.surface_distance.metrics import *

def dice_coefficient(truth, prediction):
    return 2 * np.sum(truth * prediction)/(np.sum(truth) + np.sum(prediction))

def calc_dice_per_slice(test_truth, prediction_filled):
    dice_per_slice_dict = {}
    num_slices = test_truth.shape[2]

    for i in range(0,num_slices):
        indices_truth = np.nonzero(test_truth[:,:,i]>0)
        indices_pred = np.nonzero(prediction_filled[:,:,i]>0)
        if ((len(indices_truth[0])) == 0 & (len(indices_pred[0]) == 0)):
            continue
        dice_per_slice = dice(test_truth[:,:,i], prediction_filled[:,:,i])
        dice_per_slice_dict[i+1] = dice_per_slice

    return dice_per_slice_dict

def calc_overlap_measure_per_slice(truth, prediction, eval_function):
    eval_per_slice_dict = {}
    num_slices = truth.shape[2]

    for i in range(0,num_slices):
        #evaluate only slices that have at least one truth pixel or predction pixel
        indices_truth = np.nonzero(truth[:,:,i]>0)
        indices_pred = np.nonzero(prediction[:,:,i]>0)
        if ((len(indices_truth[0])) == 0 and (len(indices_pred[0]) == 0)):
            continue

        eval_per_slice = eval_function(truth[:, :, i], prediction[:, :, i])
        eval_per_slice_dict[i+1] = eval_per_slice

    return eval_per_slice_dict


def IoU(gt_seg, estimated_seg):
    """
    compute Intersection over Union
    :param gt_seg:
    :param estimated_seg:
    :return:
    """
    seg1 = np.asarray(gt_seg).astype(np.bool)
    seg2 = np.asarray(estimated_seg).astype(np.bool)

    # Compute IOU
    intersection = np.logical_and(seg1, seg2)
    union = np.logical_or(seg1, seg2)

    return intersection.sum() / union.sum()


def VOE(gt_seg, pred_seg):
    """
    compute volumetric overlap error (in percent) = 1 - intersection/union
    :param gt_seg:
    :param pred_seg:
    :return:
    """
    return 1 - IoU(gt_seg, pred_seg)


def seg_ROI_overlap(gt_seg, roi_pred):
    """
    compare ground truth segmentation to predicted ROI, return number of voxels from gt seg that aren't contained in
    the predicted ROI
    :param gt_seg: segmentation
    :param roi_pred: ROI represented as a binary segmentation
    :return:
    """
    seg = np.asarray(gt_seg).astype(np.bool)
    seg_roi = np.asarray(roi_pred).astype(np.bool)

    # if segmentation is bigger than intersection seg_roi, we are out of bounds
    intersection = np.logical_and(seg, seg_roi)
    return np.sum(seg ^ intersection)


def vod(mask1, mask2, verbose=False):
    mask1, mask2 = mask1.flatten(), mask2.flatten()
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if verbose:
        print('intersection\t', intersection)
        print('union\t\t', union)
    return 1 - (intersection + 1) / (union + 1)


def dice(y_true, y_pred, smooth=1.):
    y_true_f = y_true.flatten() > 0
    y_pred_f = y_pred.flatten() > 0
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)


def nvd(mask1, mask2):
    mask1, mask2 = mask1.flatten(), mask2.flatten()
    sum_ = (mask1.sum() + mask2.sum())
    diff = abs(mask1.sum()-mask2.sum())
    return 2 * diff / sum_


def hosdorf_and_assd(y_true, y_pred, scaling):
    surface_distances = compute_surface_distances(y_true, y_pred, spacing_mm=scaling)
    assd = np.mean(compute_average_surface_distance(surface_distances))
    hausdorff = compute_robust_hausdorff(surface_distances, 100)
    return hausdorff, assd


def hausdorff(y_true, y_pred, scaling):
    surface_distances = compute_surface_distances(y_true, y_pred, spacing_mm=scaling)
    #assd = np.mean(surface_distance.compute_average_surface_distance(surface_distances))
    hausdorff = compute_robust_hausdorff(surface_distances, 95)
    return hausdorff


def assd(y_true, y_pred, scaling):
    surface_distances = compute_surface_distances(y_true, y_pred, spacing_mm=scaling)
    assd = np.mean(compute_average_surface_distance(surface_distances))
    return assd