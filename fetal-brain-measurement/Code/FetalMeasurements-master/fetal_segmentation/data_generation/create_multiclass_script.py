from utils.read_write_data import *
import glob


def dict_from_pathes(reference_pathes, num_labels):
    dirnames_dict = {}
    for path in reference_pathes:
        dirname = os.path.basename(path)
        labels_list = [None]*num_labels
        labels_list[0] = path
        dirnames_dict[dirname] = labels_list

    return dirnames_dict


def get_shared_labels(labels_data_path):
    num_labels = len(labels_data_path)

    reference_pathes = glob.glob(os.path.join(labels_data_path[0], '*'))
    dirnames_dict = dict_from_pathes(reference_pathes, num_labels)

    for i in range(1, num_labels):
        pathes = glob.glob(os.path.join(labels_data_path[1], '*'))
        for path in pathes:
            dir = os.path.basename(path)
            if(dir in dirnames_dict):
                labels_list = dirnames_dict[dir]
                labels_list[i] = path

    return dirnames_dict




"""
This script unifies labels to one labels nifti file
Assumes that folder_ids can be matched
"""
if __name__ == '__main__':
    labels_data_path = {}
    my_path = os.path.abspath(os.path.dirname(__file__))
    labels_data_path[0] = '/home/bella/Phd/data/fetal_mr/placenta/raw_data/'
    labels_data_path[1] = '/home/bella/Phd/data/fetal_mr/body/raw_data/'
    vol_data_path = '/home/bella/Phd/data/fetal_mr/body/raw_data/'
    unified_truth_save_path = '/home/bella/Phd/data/fetal_mr/placenta_body_unified/test/'
    num_labels = len(labels_data_path)

    dirnames_dict = get_shared_labels(labels_data_path)
    for dirname, truth_pathes in dirnames_dict.items():
        path = os.path.join(truth_pathes[0], 'truth.nii')
        truth, affineData, hdr= read_nifti_vol_meta(path)
        vol_size = truth.shape
        unified_truth = np.zeros(vol_size, dtype=np.uint8)

        for i in range(0,num_labels):
            path = os.path.join(truth_pathes[i], 'truth.nii')
            truth = read_nifti(path)
            nonzero_indices = np.nonzero(truth)
            unified_truth[nonzero_indices] = i+1

        save_dir = os.path.join(unified_truth_save_path, dirname)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        truth_save_path = os.path.join(save_dir,'truth.nii')
        save_to_nifti(unified_truth, affineData, hdr, truth_save_path)

        vol_path = os.path.join(vol_data_path, dirname, 'volume.nii')
        vol_save_path = os.path.join(unified_truth_save_path, dirname, 'volume.nii')
        copyfile(vol_path, vol_save_path)



