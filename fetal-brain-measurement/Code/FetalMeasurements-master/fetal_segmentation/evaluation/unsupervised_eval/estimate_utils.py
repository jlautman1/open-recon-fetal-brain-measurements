from evaluation.eval_utils.eval_functions import calc_overlap_measure_per_slice, dice
import numpy as np


def calc_mean_dice_per_slice(curr_pred_dice, num_slices):
    dice_per_slice_dict = {}
    num_tta = len(curr_pred_dice)
    for i in range(1,num_slices+1):
        vals = []
        for j in range(num_tta):
            if(i in curr_pred_dice[j]):
                vals.append(curr_pred_dice[j][i])
        if(len(vals)>0):
            dice_per_slice_dict[i] = np.mean(vals)

    return dice_per_slice_dict


def estimate_overlap_measure_per_slice(tta_preds, num_slices):
    mean_dices = []
    for i in range(len(tta_preds)):
        curr_pred = tta_preds[i]
        curr_pred_dice = []
        for j in range(len(tta_preds)):
            if(i==j):
                continue
            curr_pred_dice.append(calc_overlap_measure_per_slice(curr_pred, tta_preds[j], dice))
        dice_per_slice_dict = calc_mean_dice_per_slice(curr_pred_dice, num_slices)
        mean_dices.append(dice_per_slice_dict)

    mean_all = calc_mean_dice_per_slice(mean_dices, num_slices)
    return mean_all