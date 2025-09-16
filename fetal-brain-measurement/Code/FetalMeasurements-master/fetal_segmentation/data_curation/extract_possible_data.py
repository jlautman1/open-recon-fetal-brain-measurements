import pandas as pd
import os
from shutil import copyfile


if __name__ == "__main__":
    csv_path = '/home/bella/Phd/data/cor_FRFSE.csv'
    data_path = '/media/bella/8A1D-C0A6/Phd/data/data_all/'
    outdir = '/media/bella/8A1D-C0A6/Phd/data/Brain/FR-FSE/'

    df = pd.read_csv(csv_path)
    dir_names = df['Filename'].tolist()

    for filename in dir_names:
        splitted_path = os.path.split(filename)
        filename = splitted_path[-1]
        dirname = os.path.split(splitted_path[0])[-1]
        search_path = os.path.join(data_path, dirname, filename)
        if(os.path.isfile(search_path)):
            out_path = os.path.join(outdir, splitted_path[-1])
            copyfile(search_path, out_path)