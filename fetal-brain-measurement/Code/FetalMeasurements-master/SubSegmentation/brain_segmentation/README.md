# Segmentation Model

In order to run on new scan:

```
model = BrainSegmentationModel((160, 160), 'path_to_model', 'model_name')
img = nib.load('path_to_scan').get_fdata()
model.predict_nifti(img, os.path.join('dest_directory_path', 'dest_filename'), tta=<True/False>)
```
