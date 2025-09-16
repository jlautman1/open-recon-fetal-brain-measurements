import nibabel as nib
import numpy as np
import pickle
import os
import ntpath
from shutil import copyfile


def save_nifti(data, path):
    nifti = get_image(data)
    nib.save(nifti, path)

def save_nifti_pred(data, out_path, reference_img_path):
    """
    Save a prediction volume so it lines-up with the original scan.

    Parameters
    ----------
    data : np.ndarray            # your label / probability map
    out_path : str               # where to write the .nii.gz
    reference_img_path : str     # the *source* anatomical image
    """
    ref   = nib.load(reference_img_path)
    hdr   = ref.header.copy()

    # Make sure the datatype is something tiny (labels → uint8 or int16)
    hdr.set_data_dtype(np.uint8 if data.max() < 256 else np.int16)

    # Copy spatial information
    code = int(ref.header.get('sform_code', 1))
    hdr.set_sform(ref.affine, code=code or 1)
    hdr.set_qform(ref.affine, code=int(ref.header.get('qform_code', 1)))

    # (optional but nice – lets viewers scale the overlay colours properly)
    hdr['cal_min'] = 0
    hdr['cal_max'] = data.max()

    out_img = nib.Nifti1Image(np.asarray(data), ref.affine, hdr)
    nib.save(out_img, out_path)


def read_img(in_file):
    image = nib.load(os.path.abspath(in_file))
    return image

def get_image(data, affine=None, nib_class=nib.Nifti1Image):
    if affine is None:
        affine = np.eye(4)
    return nib_class(dataobj=data, affine=affine)


def pickle_dump(item, out_file):
    with open(out_file, "wb") as opened_file:
        pickle.dump(item, opened_file)


def pickle_load(in_file):
    with open(in_file, "rb") as opened_file:
        return pickle.load(opened_file)

def load_nifti(in_file):
    print("Reading: {0}".format(in_file))
    image = nib.load(os.path.abspath(in_file))
    return image


def read_nifti_vol_meta(path):
    nib_vol = load_nifti(path)
    affineData = nib_vol.affine
    hdr = nib_vol.header
    npVol = nib_vol.get_data()

    return npVol, affineData, hdr


def read_nifti(path):
    nib_vol = load_nifti(path)
    npVol = nib_vol.get_data()

    return npVol

def save_data_splits(out_dir, training_file, validation_file, test_file):
    split_dir = os.path.dirname(training_file)
    split_dirname = os.path.basename(split_dir)
    out_dir = os.path.join(out_dir, split_dirname)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    training_file_out = os.path.join(out_dir, ntpath.basename(training_file))
    validation_file_out = os.path.join(out_dir, ntpath.basename(validation_file))
    test_file_out = os.path.join(out_dir, ntpath.basename(test_file))

    copyfile(training_file, training_file_out)
    copyfile(validation_file, validation_file_out)
    copyfile(test_file, test_file_out)

def save_norm_params(out_dir, data_dir):
    norm_params_path = os.path.join(data_dir, 'norm_params.json')
    out_path = os.path.join(out_dir, 'norm_params.json')
    copyfile(norm_params_path, out_path)


def save_old_model(out_dir, old_model_path):
    model_basename = os.path.basename(old_model_path)
    out_path = os.path.join(out_dir,model_basename)
    copyfile(old_model_path, out_path)


def save_to_nifti(np_array, affine, header, out_path):
    nifti_prediction = nib.nifti1.Nifti1Image(np_array, affine, header)
    nib.save(nifti_prediction, out_path)


def list_dump(l, out_file):
    np.savetxt(out_file, l, fmt='%s')


def list_load(in_file):
    return list(np.loadtxt(in_file, dtype=str, ndmin=1))