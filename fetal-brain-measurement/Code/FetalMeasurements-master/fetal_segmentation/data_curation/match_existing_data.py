import os
import pandas as pd
from data_curation.helper_functions import read_ids

if __name__ == "__main__":
    ids_dir = '/home/bella/Phd/data/body/FIESTA/'
    data_description_path = '/home/bella/Phd/data/body/server_data_description/'
    unverified_list_path = data_description_path + 'dafi_gt.csv'
    annotator1_list_path = data_description_path + 'Liat_gt.csv'
 #   annotator2_list_path = data_description_path + 'Elka_gt.csv'

    ids = os.listdir(ids_dir)

    annotator1_dict = read_ids(annotator1_list_path)
  #  annotator2_dict = read_ids(annotator2_list_path)
    unverified_dict = read_ids(unverified_list_path)

    ids_dict = {}

    for id in ids:
        match_str = ''
        if(id in annotator1_dict):
            match_str = match_str + annotator1_dict[id] + '_annotator1,'
            print('found id ' + id + ' in annotator1 directory')
        # if(id in annotator2_dict):
        #     match_str = match_str + annotator2_dict[id] + '_annotator2,'
        #     print('found id ' + id + ' in annotator2 directory')
        if(id in unverified_dict):
            match_str = match_str + unverified_dict[id] + '_Dafi'
            print('found id ' + id + ' in unverified directory')
        ids_dict[id] = match_str

    df = pd.DataFrame.from_dict(ids_dict , orient='index')
    df.to_csv(data_description_path + 'matching.csv')
