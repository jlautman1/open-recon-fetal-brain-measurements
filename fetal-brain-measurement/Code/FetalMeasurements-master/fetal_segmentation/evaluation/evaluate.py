import argparse
import pickle
from glob import glob
from pathlib import Path

import pandas as pd
import re
from astropy.stats.sigma_clipping import sigma_clipped_stats
from multiprocess.dummy import Pool
from scipy.stats import gmean
from tqdm import tqdm_notebook as tqdm

from evaluation.eval_utils.eval_functions import *
from evaluation.eval_utils.key_images import *
from evaluation.eval_utils.postprocess import postprocess_prediction

scaling = (1.48,1.48,3)


def mean(data):
    return data.mean(axis=0)
def meadian(data):
    return np.median(data, axis=0)

def rob_mean(data):
    return sigma_clipped_stats(data, axis=0, sigma=3, maxiters=1)[0]

def rob_median(data):
    return sigma_clipped_stats(data, axis=0, sigma=3, maxiters=1)[1]

def geometric_mean(data):
    return gmean(data, axis=0)

def postprocess(data):
    return postprocess_prediction(data, threshold=0.5,
                                  fill_holes=False, connected_component=True)


def get_vol_sizes(vol_ids, eval_folder):
    sizes_dict = {}

    for id in vol_ids:
        vol_path = os.path.join(eval_folder, str(id), 'data.nii.gz')
        vol = nib.load(vol_path).get_data()
        sizes_dict[id] = vol.shape

    return sizes_dict


def do_all(path_list, name='', overlap_metrics=[dice, vod, nvd], distance_metrics=[hausdorff, assd]):
    print(name)
    print('-----------------------------------------')
    print('-----------------------------------------')
    metrics = overlap_metrics + distance_metrics
    for aggr_method in [meadian]: #, mean
        print('-----------------------------------------')
        print(aggr_method.__name__, flush=True)
        pred_scores_per_slice = {skey.__name__ : {} for skey in overlap_metrics}
        pred_scores_vol = {skey.__name__ : {} for skey in metrics}

        def process_sub(subject_folder):
            subject_id = os.path.basename(subject_folder)
            id = int(subject_id)
            truth = nib.load(os.path.join(subject_folder, 'truth.nii.gz')).get_data()
            pred = nib.load(os.path.join(subject_folder, 'prediction.nii.gz'))
            if(truth.shape != pred.shape):
                print("in case + " + subject_folder + " there is a mismatch")
            pred_mean = pred.get_data() #aggr_method(pred.get_data())
            #del pred

            pred_mean_sec = postprocess_prediction(pred_mean)
            for score_method in overlap_metrics:
                pred_scores_per_slice[score_method.__name__][id] = calc_overlap_measure_per_slice(truth, pred_mean_sec, score_method)
                pred_scores_vol[score_method.__name__][id] = score_method(truth, pred_mean_sec)

            for score_method in distance_metrics:
                pred_scores_vol[score_method.__name__][id] = score_method(truth, pred_mean_sec, scaling)

            del pred_mean
            del pred_mean_sec
            del truth

            #can give to Pool() less workers as a parameter
        with Pool() as pool:
            list(tqdm(pool.imap_unordered(process_sub, path_list), total=len(path_list)))

        print('\t\t volumetric measures')
        for score_method in metrics:
            score_key = score_method.__name__
            print('{}\t - {:.3f} ±({:.3f}))'.format(score_key,
                                                np.mean(list(pred_scores_vol[score_key].values())),
                                                np.std(list(pred_scores_vol[score_key].values()))))
        return pred_scores_per_slice, pred_scores_vol


def save_to_csv(pred_scores_vol, path):
    pred_df = pd.DataFrame.from_dict(pred_scores_vol)
    pred_df.to_csv(path)


def path_to_name(path):
    basename = os.path.basename(path)
    basename = os.path.splitext(basename)[0]
    splitted = basename.split('_')
    title = 'slice ' + splitted[2] + ': dice=' + splitted[3]
    return title


def write_to_excel(pred_scores_vol, pred_dice_per_slice, excel_path, eval_folder, num_key_images, thresh_value):

    writer = pd.ExcelWriter(excel_path, engine='xlsxwriter')
    sorted_keys = sorted(pred_dice_per_slice.keys())

    #write volumetric evaluation
    df = pd.DataFrame.from_dict(pred_scores_vol)
    df = df.round(2)#round to 2 decimal digits
    df.to_excel(writer, sheet_name='vol_eval')

    #write volume sizes
    sizes_dict = get_vol_sizes(sorted_keys, eval_folder)
    df_sizes = pd.DataFrame.from_dict(sizes_dict).T
    df_sizes.to_excel(writer, sheet_name='vol_sizes')

    #write 2D evaluations in different tab for each volume
    for vol_id in sorted_keys:
        #write slice evaluation data
        vol_slice_eval = pred_dice_per_slice[vol_id]
        df_2D = pd.DataFrame.from_dict(vol_slice_eval, orient='index')
        df_2D = df_2D.round(3)
        sheet_name = 'vol_' + str(vol_id)
        df_2D.to_excel(writer, sheet_name=sheet_name)

        #write slice evaluation graph
        workbook = writer.book
        worksheet = writer.sheets[sheet_name]
        chart = workbook.add_chart({'type': 'line'})
        num_slices = len(vol_slice_eval)
        chart.add_series({
            'categories': [sheet_name, 1, 0, num_slices, 0],
            'values':     [sheet_name, 1, 1, num_slices, 1],
        })
        chart.set_x_axis({'name': 'Slice number', 'position_axis': 'on_tick'})
        chart.set_y_axis({'name': '2D dice', 'major_gridlines': {'visible': False}})
        chart.set_legend({'position': 'none'})
        worksheet.insert_chart('D2', chart)

        #write key images
        key_images_indices = get_key_slices_indexes(vol_slice_eval, num_key_images, thresh_value)
        images_pathes = save_key_images(key_images_indices, eval_folder, vol_id) #save images to load from in excel
        sorted_slices = sorted(images_pathes.keys())
        start_row = 20
        figure_hight = 18
        for slice_num in sorted_slices:
            plotname = path_to_name(images_pathes[slice_num])
            worksheet.insert_image('D' + str(start_row), images_pathes[slice_num])
            worksheet.write('F' + str(start_row-2), plotname)
            start_row = start_row + figure_hight + 1

    writer.save()


def write_dice_per_slice(dice_per_slice, save_path):
    dice_all_slices = {}

    for key in dice_per_slice:
        vol_dice = np.full(100,None, dtype=float)
        vol_dict = dice_per_slice[key]
        for slice_num in vol_dict:
          vol_dice[slice_num] = vol_dict[slice_num]
        dice_all_slices[key] = vol_dice
    pd_scores = pd.DataFrame.from_dict(dice_all_slices)
    pd_scores.to_csv(save_path)


if __name__ == '__main__':
    """
    This scripts assumes there is a directory with results in the needed format (after running predict_nifti_dir.py)
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--src_dir", help="specifies source directory for evaluation",
                        type=str, required=True)
    parser.add_argument("--num_key_imgs", help="specifies source directory for evaluation",
                        type=int, required=False, default=4)
    parser.add_argument("--dice_thresh", help="specifies source directory for evaluation",
                        type=int, required=False, default=0.92)
    opts = parser.parse_args()
    src_dir = opts.src_dir
    num_key_images = opts.num_key_imgs
    dice_thresh = opts.dice_thresh

    if Path(src_dir+'/pred_scores_per_slice.pkl').exists() and Path(src_dir+'/pred_scores_vol.pkl').exists():
        print('scores were already calculated, loading')
        with open(src_dir+'/pred_scores_per_slice.pkl', 'rb') as f:
            pred_scores_per_slice = pickle.load(f)
        with open(src_dir+'/pred_scores_vol.pkl', 'rb') as f:
            pred_scores_vol = pickle.load(f)
    else:
        pred_scores_per_slice, pred_scores_vol = do_all([_ for _ in (glob(os.path.join(src_dir, 'test', '*')))], 'test')

        print('--------------------\nsaving...')
        with open(src_dir+'/pred_scores_per_slice.pkl', 'wb') as f:
            pickle.dump(pred_scores_per_slice, f)
        with open(src_dir+'/pred_scores_vol.pkl', 'wb') as f:
            pickle.dump(pred_scores_vol, f)

    write_to_excel(pred_scores_vol, pred_scores_per_slice['dice'], src_dir + '/pred_scores.xlsx', os.path.join(src_dir, 'test'), num_key_images, dice_thresh)
    write_dice_per_slice(pred_scores_per_slice['dice'], src_dir + '/dice_per_slice.csv')