import os
from data_curation.helper_functions import read_ids
import pandas as pd


if __name__ == "__main__":
    ids_dir = '/home/bella/Phd/data/body/FIESTA/'
    data_description_path = '/home/bella/Phd/data/body/server_data_description/'
    unverified_list_path = data_description_path + 'dafi_gt.csv'
    annotator1_list_path = data_description_path + 'Liat_gt.csv'
    annotator2_list_path = data_description_path + 'Elka_gt.csv'

    ids_lst = os.listdir(ids_dir)
    ids_set = set(ids_lst)

    annotator1_dict = read_ids(annotator1_list_path)
    annotator2_dict = read_ids(annotator2_list_path)
    unverified_dict = read_ids(unverified_list_path)

    new_ids_dict = {}

    for id in annotator1_dict.keys():
        if(id not in ids_set):
            new_ids_dict[id] = annotator1_dict[id] + '_annotator1'

    for id in annotator2_dict.keys():
        if(id not in ids_set):
            if(id not in new_ids_dict):
                new_ids_dict[id] = annotator2_dict[id] + '_annotator2'
            else:
                new_ids_dict[id] = new_ids_dict[id] + ',' + annotator2_dict[id] + '_annotator2'

    for id in unverified_dict.keys():
        if(id not in ids_set):
            if(id not in new_ids_dict):
                new_ids_dict[id] = unverified_dict[id] + '_Dafi'
            else:
                new_ids_dict[id] = new_ids_dict[id] + ',' + unverified_dict[id] + '_Dafi'


    df = pd.DataFrame.from_dict(new_ids_dict , orient='index')
    df.to_csv(data_description_path + 'additional_data.csv')