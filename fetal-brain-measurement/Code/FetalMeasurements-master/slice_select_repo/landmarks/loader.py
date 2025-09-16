### Loader for 3D Nifti and metadata

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch._utils import _accumulate
from torch import randperm
from torchvision import transforms, utils
import nibabel
import os
import numpy as np
import itertools


def bbox_ND(img):
    N = img.ndim
    out = []
    for ax in itertools.combinations(reversed(range(N)), N - 1):
        nonzero = np.any(img, axis=ax)
        out.append(np.where(nonzero)[0][[0, -1]])

    return tuple(out)


class NiftiLandmarksDataset(Dataset):
    """Nifti dataset."""

    def __init__(self, csv_file, nii_dir, seg_dir, transform=None, only_tag=True, tagname="TCD_Selection", filter_quality=4):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            only_tag (bool): Take only samples with GT
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        df = pd.read_excel(csv_file)
        self.nii_dir = nii_dir
        self.seg_dir = seg_dir
        self.transform = transform

        #Filter metadata
        if filter_quality is not None:
            df = df.drop(df[df.QualLevel < filter_quality].index).reset_index()
        if only_tag:
            df = df.dropna(subset=[tagname, ]).reset_index()

        df = df.assign(image=np.nan).assign(seg_image=np.nan)
        df.image = df.image.astype(object)
        df.seg_image = df.seg_image.astype(object)

        df = df.assign(bbox=np.nan)
        df.bbox = df.bbox.astype(object)

        df.Subject = df.Subject.astype(int)
        df.Series = df.Series.astype(int)

        df.resZ = df.resZ.astype(int)



        self.metadata = df

    def __len__(self):
        return len(self.metadata) - 1

    def __getitem__(self, idx):

        # Load Nifti
        if isinstance(self.metadata.at[idx, "image"], float):
            filename = "Pat{patient_id:02}_Se{series:02}_Res{res_x}_{res_y}_Spac{res_z}.nii"
            fn = filename.format(patient_id=self.metadata.loc[idx, "Subject"],
                                 series=self.metadata.loc[idx, "Series"],
                                 res_x=self.metadata.loc[idx, "resX"],
                                 res_y=self.metadata.loc[idx, "resY"],
                                 res_z=self.metadata.loc[idx, "resZ"],)
                    
            img_name = os.path.join(self.nii_dir, fn)
            image = nibabel.load(img_name).get_fdata().astype(np.float)
            image = image.transpose([2,0,1])
            self.metadata.at[idx, "image"] = image
            self.metadata.at[idx, "filename"] = img_name

        # Load BBox
        if isinstance(self.metadata.at[idx, "seg_image"], float):
            filename = "Pat{patient_id:02}_Se{series:02}_Res{res_x}_{res_y}_Spac{res_z}_roi_pred_pp.nii.gz"
            fn = filename.format(patient_id=self.metadata.loc[idx, "Subject"],
                                 series=self.metadata.loc[idx, "Series"],
                                 res_x=self.metadata.loc[idx, "resX"],
                                 res_y=self.metadata.loc[idx, "resY"],
                                 res_z=self.metadata.loc[idx, "resZ"],)
                    
            bbox_img_name = os.path.join(self.seg_dir, fn)
            bbox_image = nibabel.load(bbox_img_name).get_fdata().astype(np.float)
            bbox_image = bbox_image.transpose([2,0,1])
            bbox_image = bbox_image > .5
            
            self.metadata.at[idx, "seg_image"] = bbox_image
            self.metadata.at[idx, "seg_filename"] = bbox_img_name
            
            bbox_idxs = bbox_ND(bbox_image)
            bbox_idxs = (np.array([0,bbox_image.shape[0]]), *bbox_idxs[1:])

            self.metadata.at[idx, "bbox"] = bbox_idxs

        # Load metadata
        sample = self.metadata.loc[idx].to_dict()

        # Hack fixing selection
        for k in sample.keys():
            if "Selection" in k:
                sample[k] = sample[k] - 1

        # Load measures
        sample["landmarks"] = np.array([ [sample["Measure_TCD_X1"], sample["Measure_TCD_Y1"], ],
                                         [sample["Measure_TCD_X2"], sample["Measure_TCD_Y2"], ],] )

        if self.transform:
            sample = self.transform(sample)

        return sample

    def get_metadata(self, idx):
        if not isinstance(idx, list):
            idx = [idx, ]
        idx = np.array(idx)
        return self.metadata.loc[idx]

def random_split(dataset, lengths):
    """
    Randomly split a dataset into non-overlapping new datasets of given lengths.

    Arguments:
        dataset (Dataset): Dataset to be split
        lengths (sequence): lengths of splits to be produced
    """
    if sum(lengths) != len(dataset):
        raise ValueError("Sum of input lengths does not equal the length of the input dataset!")

    indices = randperm(sum(lengths)).tolist()
    ret_ds = []
    ret_indices = []
    for offset, length in zip(_accumulate(lengths), lengths):
        cur_ds_indices = indices[offset - length:offset]
        ret_ds.append(torch.utils.data.Subset(dataset, cur_ds_indices))
        ret_indices.append(cur_ds_indices)
    return ret_ds, ret_indices

if __name__ == '__main__':
    brain_dataset = NiftiDataset(csv_file='/media/df3-dafna/Netanell/BLLAMODEL/Data.xlsx',
                                 root_dir='/media/df3-dafna/Netanell/BLLAMODEL/')
    for i in range(len(brain_dataset)):
        sample = brain_dataset[i]
        print(i, sample['image'].shape, sample['TCD_Selection'])