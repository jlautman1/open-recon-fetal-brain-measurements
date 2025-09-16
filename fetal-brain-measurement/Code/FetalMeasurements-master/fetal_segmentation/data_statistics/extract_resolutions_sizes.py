import argparse
import glob
import os
import nibabel as nib
import pandas as pd
import re
from struct import *


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", help="specifies nifti file dir path",
                        type=str, required=True)
    parser.add_argument("--output_folder", help="specifies nifti file dir path",
                        type=str, required=True)
    return parser.parse_args()


def get_scan_data(filename, scan_re):
    res = scan_re.findall(filename)
    pat_id, ser_num, res_x, res_y,  res_z = res[0]
    vol = nib.load(filename).get_data()
    x_size, y_size, z_size = vol.shape
    return pat_id, res_x, res_y, res_z, x_size, y_size, z_size


if __name__ == '__main__':

    opts = get_arguments()

    filenames = glob.glob(os.path.join(opts.input_dir, '*'))

    scan_re = re.compile("Pat(?P<patient_id>[\d]+)_Se(?P<series>[\d]+)_Res(?P<res_x>[\d.]+)_(?P<res_y>[\d.]+)_Spac(?P<res_z>[\d.]+).nii")

    data = []

    for filename in filenames:
        pat_id, res_x, res_y, res_z, x_size, y_size, z_size = get_scan_data(filename, scan_re)
        data.append([pat_id, res_x, res_y, res_z, x_size, y_size, z_size])

    columns = ['scan_id','x_res', 'y_res', 'z_res', 'x_size', 'y_size', 'z_size']
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(opts.output_folder + 'resolutions_sizes.csv')






