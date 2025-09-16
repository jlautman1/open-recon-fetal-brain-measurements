import argparse

from evaluation.eval_utils.prediction import predict_case, get_prediction_params
from training.train_functions.training import *
from utils.read_write_data import *


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_nii", help="specifies mat file dir path",
                        type=str, required=True)
    parser.add_argument("--output_folder", help="specifies mat file dir path",
                        type=str, required=True)
    # Params for primary prediction
    parser.add_argument("--config_dir", help="specifies config dir path",
                        type=str, required=True)

    parser.add_argument("--overlap_factor", help="specifies overlap between prediction patches",
                        type=float, default=0.7)
    parser.add_argument("--preprocess", help="what preprocess to do",
                        type=str, required=False, default=None)

    opts = parser.parse_args()
    return opts


def get_vol_id(input_path):
    dir_path = os.path.dirname(input_path)
    return os.path.basename(dir_path)


if __name__ == '__main__':

    opts = get_arguments()

    input_path = opts.input_nii
    output_path = opts.output_folder
    overlap_factor = opts.overlap_factor
    preprocess = opts.preprocess
    vol_id = get_vol_id(input_path)

    config, norm_params, model_path = get_prediction_params(opts.config_dir)

    print(model_path)
    model = load_old_model(get_last_model_path(model_path))
    prediction, nifti = predict_case(model, input_path, config['patch_shape'], config['patch_depth'], preprocess, norm_params, overlap_factor)

    nifti_prediction = nib.nifti1.Nifti1Image(prediction, nifti.affine, nifti.header)
    nib.save(nifti_prediction, os.path.join(output_path, 'prediction_' + vol_id + '.nii'))
    nib.save(nifti, os.path.join(output_path, 'vol_' + vol_id + '.nii'))
