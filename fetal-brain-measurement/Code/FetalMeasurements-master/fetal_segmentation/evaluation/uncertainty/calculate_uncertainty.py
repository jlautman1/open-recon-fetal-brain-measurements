import argparse
import os
import glob
import nibabel as nib
from evaluation.uncertainty.entropy import *
from utils.visualization import plot_data


def get_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", help="specifies TTA dir path",
                        type=str, required=True)
    parser.add_argument("--output_dir", help="specifies output path",
                        type=str, required=True)

    opts = parser.parse_args()

    return opts.input_dir, opts.output_dir


if __name__ == '__main__':

    input_dir, output_dir = get_arguments()

    scan_dirs = glob.glob(os.path.join(input_dir, '*'))

    for scan_dir in scan_dirs:
        volume = nib.load(os.path.join(scan_dir,'data.nii.gz')).get_fdata()
        predictions = nib.load(os.path.join(scan_dir,'predictions.nii.gz')).get_fdata()
        ent = entropy(predictions)

        plot_data(volume[:,:,30],ent[:,:,30], 'uncertainty')


