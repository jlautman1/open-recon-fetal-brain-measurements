import nibabel as nib
import os


input_path = '/home/bella/Phd/data/fetal_mr/body/TRUFI_body/Pat331_Se22_Res1.25_1.25_Spac3.5.nii'

nifti_data = nib.load(os.path.abspath(input_path))
vol_data = nifti_data.get_fdata()