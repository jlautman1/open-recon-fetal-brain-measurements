import argparse
import json
import os

from evaluation.eval_utils.prediction import evaluate_cases, get_prediction_params


def get_eval_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_dir", help="specifies config dir path",
                        type=str, required=True)
    parser.add_argument("--split", help="What split to predict on? (test/val)",
                        type=str, default='test')
    parser.add_argument("--overlap_factor", help="specifies overlap between prediction patches",
                        type=float, default=0.75)
    parser.add_argument("--preprocess", help="specifies preprocessing method",
                        type=str, default=None)
    opts = parser.parse_args()

    return opts


def run_eval(config, config_dir, norm_params, preprocess =None, split='test', overlap_factor=0.7):
    prediction_dir = os.path.abspath(os.path.join(config_dir, 'predictions', split))
    my_path = os.path.abspath(os.path.dirname(__file__))

    indices_file = {
      "test": os.path.join(config_dir, "debug_split/test_ids.pkl"),
      "val": os.path.join(config_dir, "debug_split/test_ids.pkl"),
      "train": os.path.join(config_dir, "debug_split/test_ids.pkl")
    }[split]
    data_dir = os.path.join(my_path, config["data_dir"])
    scans_dir = os.path.join(my_path, config["scans_dir"])

    model_file = os.path.join(config_dir, config["model_pref"])
    evaluate_cases(indices_file, model_file, hdf5_file=os.path.join(data_dir, "fetal_data.h5"),
                   patch_shape=config["patch_shape"], patch_depth=config["patch_depth"], output_dir=prediction_dir,
                   raw_data_path=scans_dir, preprocess=preprocess, norm_params = norm_params, overlap_factor=overlap_factor)

if __name__ == "__main__":
    """
    This script evaluates dice per volume and dice per slice and writes a csv file for each volume and a summery csv
    """
    opts = get_eval_arguments()

    with open(os.path.join(opts.config_dir, 'config.json')) as f:
        config = json.load(f)

    config, norm_params, model_path = get_prediction_params(opts.config_dir, normalize=False)

    run_eval(config, opts.config_dir, norm_params, preprocess=opts.preprocess, split = opts.split, overlap_factor = opts.overlap_factor)