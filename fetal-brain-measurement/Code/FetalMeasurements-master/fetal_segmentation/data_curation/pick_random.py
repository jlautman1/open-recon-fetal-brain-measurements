import random
import os
from shutil import copyfile


if __name__ == "__main__":
    data_path = '/home/bella/Phd/data/brain/TRUFI_all/'
    out_path = '/home/bella/Phd/data/brain/TRUFI_sample/'
    num_to_select = 20
    filenames = os.listdir(data_path)
    picked_samples = set(random.sample(filenames, num_to_select))

    for filename in picked_samples:
        src_path = os.path.join(data_path,filename)
        dst_path = os.path.join(out_path, filename)
        copyfile(src_path, dst_path)