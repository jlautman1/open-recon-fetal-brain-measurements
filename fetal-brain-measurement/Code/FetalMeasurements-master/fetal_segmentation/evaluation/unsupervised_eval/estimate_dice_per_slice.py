import argparse
import glob
import os
import nibabel as nib
from multiprocess.dummy import Pool
from scipy.stats import gmean
from tqdm import tqdm_notebook as tqdm
import numpy as np
from evaluation.eval_utils.eval_functions import calc_overlap_measure_per_slice, dice
from evaluation.unsupervised_eval.estimate_utils import estimate_overlap_measure_per_slice
from evaluation.eval_utils.postprocess import postprocess_prediction
import pandas as pd
from shutil import copyfile


def get_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction_dir", help="specifies prediction dir path",
                        type=str, required=True)
    parser.add_argument("--tta_dir", help="specifies TTA dir path",
                        type=str, required=True)
    parser.add_argument("--output_dir", help="specifies output path",
                        type=str, required=True)

    opts = parser.parse_args()

    return opts.prediction_dir, opts.tta_dir, opts.output_dir


def copy_preds_to_tta(tta_path_list, pred_path_list):
    #convert predictions path list to dictionary with patient id
    pred_path_dict = {}
    for path in pred_path_list:
        id = os.path.basename(path)
        pred_path_dict[id] = path

    for path in tta_path_list:
        id = os.path.basename(path)
        input_path = os.path.join(pred_path_dict[id], 'prediction.nii.gz')
        out_path = os.path.join(path, 'prediction.nii.gz')
        copyfile(input_path, out_path)


def get_intersection_slices(pred_dice_per_slice, estimated_dice_per_slice):
    """
    Get a set of non-zero slices of both in true dice calculation and estimated dice
    :param pred_dice_per_slice:
    :param estimated_dice_per_slice:
    :return: unified slice ids
    """
    pred_ids = set(pred_dice_per_slice.keys())
    estimated_ids = set(estimated_dice_per_slice.keys())
    unified_ids = pred_ids.intersection(estimated_ids)

    return unified_ids


def get_unified_slices(pred_dice_per_slice, estimated_dice_per_slice):
    """
    Get a set of non-zero slices of either in true dice calculation or estimated dice
    :param pred_dice_per_slice:
    :param estimated_dice_per_slice:
    :return: unified slice ids
    """
    pred_ids = set(pred_dice_per_slice.keys())
    estimated_ids = set(estimated_dice_per_slice.keys())
    unified_ids = pred_ids.union(estimated_ids)

    return unified_ids


def unify_dicts(pred_dice_per_slice, estimated_dice_per_slice):
    """
    Create a unified dictionary  of true and estimated dice for dataframe creation
    :param pred_dice_per_slice:
    :param estimated_dice_per_slice:
    :return: unified dict
    """
    unified_ids = get_unified_slices(pred_dice_per_slice, estimated_dice_per_slice)

    res_dict = {}
    pred = []
    estimated = []
    for id in unified_ids:
        if(id in pred_dice_per_slice):
            pred.append(pred_dice_per_slice[id])
        else:
            pred.append(None)
        if(id in estimated_dice_per_slice):
            estimated.append(estimated_dice_per_slice[id])
        else:
            estimated.append(None)

    res_dict['ids'] = unified_ids
    res_dict['true'] = pred
    res_dict['estimated'] = estimated

    return res_dict


def get_scatter_data(pred_dice_per_slice, estimated_dice_per_slice):
    scatter_points = []

    for vol_id in pred_dice_per_slice:
        intersect_ids = get_intersection_slices(pred_dice_per_slice[vol_id], estimated_dice_per_slice[vol_id])
        for slice_num in intersect_ids:
            scatter_points.append([pred_dice_per_slice[vol_id][slice_num],estimated_dice_per_slice[vol_id][slice_num]])

    return scatter_points


def write_estimation_to_excel(pred_dice_per_slice, estimated_dice_per_slice, excel_path):

    writer = pd.ExcelWriter(excel_path, engine='xlsxwriter')
    sorted_keys = sorted(pred_dice_per_slice.keys())

    scatter_data = get_scatter_data(pred_dice_per_slice, estimated_dice_per_slice)

    workbook = writer.book
    sheet_name = 'scatter_dice'
    df_scatter = pd.DataFrame(scatter_data)
    df_scatter = df_scatter.round(3)
    df_scatter.to_excel(writer, sheet_name=sheet_name)

    worksheet = writer.sheets[sheet_name]
    chart_scatter = workbook.add_chart({'type': 'scatter'})
    chart_scatter.add_series({
        'categories': [sheet_name, 1, 1, len(scatter_data), 1],
        'values':     [sheet_name, 1, 2, len(scatter_data), 2],
    })
    chart_scatter.set_x_axis({'name': 'true dice', 'position_axis': 'on_tick'})
    chart_scatter.set_y_axis({'name': 'estimated dice'})
    chart_scatter.set_legend({'position': 'none', 'major_gridlines': {'visible': False}})
    worksheet.insert_chart('D2', chart_scatter)

    #write 2D evaluations in different tab for each volume
    for vol_id in sorted_keys:
        #write slice evaluation data
        unified_dict = unify_dicts(pred_dice_per_slice[vol_id], estimated_dice_per_slice[vol_id])
        df_2D = pd.DataFrame.from_dict(unified_dict, orient='index').T
        df_2D = df_2D.round(3)
        sheet_name = 'vol_' + str(vol_id)
        df_2D.to_excel(writer, sheet_name=sheet_name)

        #write slice evaluation graph
        workbook = writer.book
        worksheet = writer.sheets[sheet_name]
        chart1 = workbook.add_chart({'type': 'line'})
        num_slices = len(unified_dict['ids'])
        chart1.add_series({
            'categories': [sheet_name, 1, 1, num_slices, 1],
            'values':     [sheet_name, 1, 3, num_slices, 3],
        })

        chart2 = workbook.add_chart({'type': 'line'})
        chart2.add_series({
            'categories': [sheet_name, 1, 1, num_slices, 1],
            'values':     [sheet_name, 1, 2, num_slices, 2],
        })
        chart1.combine(chart2)

        chart1.set_x_axis({'name': 'Slice number', 'position_axis': 'on_tick'})
        chart1.set_y_axis({'name': '2D dice', 'major_gridlines': {'visible': False}})
        chart1.set_legend({'position': 'none'})

        worksheet.insert_chart('D2', chart1)

    writer.save()


def calc_dice_per_slice_all(path_list):
    print('-----------------------------------------')
    print('calculating estimated vs. true dice-per-slice')

    pred_scores_per_slice = {}
    estimated_scores_per_slice = {}

    def process_sub(subject_folder):
        subject_id = os.path.basename(subject_folder)
        id = int(subject_id)
        truth = nib.load(os.path.join(subject_folder, 'truth.nii.gz')).get_data()
        pred = nib.load(os.path.join(subject_folder, 'prediction.nii.gz')).get_data()
        tta_preds = nib.load(os.path.join(subject_folder, 'predictions.nii.gz')).get_fdata()
        tta_preds = postprocess_prediction(tta_preds)

        if(truth.shape != pred.shape):
            print("in case + " + subject_folder + " there is a mismatch")

        pred_scores_per_slice[id] = calc_overlap_measure_per_slice(truth, pred, dice)
        estimated_scores_per_slice[id] = estimate_overlap_measure_per_slice(tta_preds, truth.shape[2])

        del pred
        del tta_preds
        del truth


    with Pool() as pool:
        list(tqdm(pool.imap_unordered(process_sub, path_list), total=len(path_list)))

    return pred_scores_per_slice, estimated_scores_per_slice


if __name__ == '__main__':

    prediction_dir, tta_dir, output_dir = get_arguments()
    pred_path_list = glob.glob(os.path.join(prediction_dir,  '*'))
    tta_path_list = glob.glob(os.path.join(tta_dir,  '*'))
    copy_preds_to_tta(tta_path_list, pred_path_list)
    pred_scores_per_slice, estimated_scores_per_slice = calc_dice_per_slice_all(tta_path_list)

    write_estimation_to_excel(pred_scores_per_slice, estimated_scores_per_slice, os.path.join(output_dir, 'dice_per_slice_estimation.xlsx'))
