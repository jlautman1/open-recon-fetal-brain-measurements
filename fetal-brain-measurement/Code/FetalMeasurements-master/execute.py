from os.path import join
import numpy as np
import nibabel as nib
import torch

from matplotlib import pyplot as plt
import os
import json
import glob
import tensorflow as tf

from SubSegmentation.lovasz import *
from SubSegmentation.processing_utils import *
import fetal_measure
import pandas as pd
import os
import json
import traceback

import argparse

import sys
sys.path.append('/workspace/fetal-brain-measurement/Code/FetalMeasurements-master/SubSegmentation')

from lovasz import lovasz_softmax
from processing_utils import acc_no_bg


def load_model():
    IMAGE_SIZE = (160, 160)
    MODEL_NAME = 'model-tfms'
    #torch.cuda.set_device(0) #not relevant in new version
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    #config = tf.ConfigProto()
    #config.gpu_options.allow_growth=True
    #sess = tf.Session(config=config)
    #got error because removed in new version

    fm = fetal_measure.FetalMeasure(subseg_model='model-tfms')
    return fm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='fetal_measure',
        description='Fetal brain MRI linear measurements')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-f', '--infile', help='Input Nifti File to execute on')
    group.add_argument('-i', '--inputdir', help='Input directory')
    parser.add_argument('-o', '--outputdir', help='Output directory', required=True)
    args = parser.parse_args()

    if args.infile is not None:
        infiles = [os.path.abspath(args.infile), ]
    elif args.inputdir is not None:
        infiles = list(glob.glob(os.path.join(args.inputdir, "*.nii*")))
    else:
        raise ValueError('Missing input dir or files')

    print("in execute file - loading model - ")
    fm = load_model()
    print("infiles are : ", infiles)
    for a in infiles:
        # a is the input file path, out_dir is the output directory path.
        print("a is: ", a)
        print("args.outputdir: ", args.outputdir)
        print("os.path.basename(a): ", os.path.basename(a))
        out_base = os.path.splitext(os.path.splitext(os.path.basename(a))[0])[0]
        out_dir = os.path.join(args.outputdir, out_base)
        try:
            fm.execute(a, out_dir)
        except Exception:
            print ('Error in file:', a)
            print(traceback.format_exc())
            continue
