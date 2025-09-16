import re
import pandas as pd
import os
import numpy as np


def read_ids(path):

    patient_ids = dict()
    df = pd.read_csv(path)
    dir_names = df['0'].tolist()

    p = re.compile("Fetus(?P<patient_id>[\d]+)")

    for name in dir_names:
        patient_id = p.findall(name)

        patient_ids[patient_id[0]] = name

    return patient_ids


def origin_id_from_filepath(filename):
    removed_extension = os.path.splitext(filename)[0]
    removed_extension = os.path.splitext(removed_extension)[0]
    basename = os.path.basename(removed_extension)
    return basename

def id_from_filepath(filename):
    basename = os.path.basename(filename)
    p = re.compile("Pat(?P<patient_id>[\d]+)_Se(?P<series_id>[\d]+)")
    ids = p.findall(basename)[0]
    patient_id = ids[0]
    series_id = ids[1]
    return patient_id + '_' + series_id


def patient_id_from_filepath(filename):
    basename = os.path.basename(filename)
    p = re.compile("Pat(?P<patient_id>[\d]+)_Se(?P<series_id>[\d]+)")
    ids = p.findall(basename)[0]
    patient_id = ids[0]
    return patient_id


def move_smallest_axis_to_z(vol):
    shape = vol.shape
    min_index = shape.index(min(shape))

    if(min_index != 2):
        vol = np.swapaxes(vol, min_index, 2)

    return vol, min_index


def swap_to_original_axis(swap_axis, vol):
    if(swap_axis != 2):
        new_vol = np.swapaxes(vol, swap_axis, 2)
        return new_vol
    return vol