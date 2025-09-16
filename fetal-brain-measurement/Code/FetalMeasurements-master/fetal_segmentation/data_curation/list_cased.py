import os
import pandas as pd

path = "/media/bella/8A1D-C0A6/Phd/data/brain_segmentations/Segmented_by_Elka/"
csv_path = '/home/bella/Phd/data/brain/server_data_description/Elka_gt.csv'
files = os.listdir(path)

df = pd.DataFrame(files)
df.to_csv(csv_path)