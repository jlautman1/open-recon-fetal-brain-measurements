import pandas as pd
import numpy as np
import math
from sklearn.ensemble import IsolationForest
import os


def middle_points(dice_per_slice):
    pts = {}
    for key in dice_per_slice:
        points = dice_per_slice[key]

        for i in range(5, len(points)-5):
            if(math.isnan(points[i]) or math.isnan(points[i-1]) or math.isnan(points[i+1]) or math.isnan(points[i-2]) or math.isnan(points[i+2]) or math.isnan(points[i-3]) or math.isnan(points[i+3]) or math.isnan(points[i-4]) or math.isnan(points[i+4]) or math.isnan(points[i-5]) or math.isnan(points[i+5])):
                continue
            # point_features = [points[i], points[i-1], points[i-2], points[i+1], points[i+2],
            #                   np.sign(points[i]-points[i-1]), np.sign(points[i]-points[i-2]), np.sign(points[i]-points[i+1]), np.sign(points[i]-points[i+2])]
            point_features = [points[i], np.square(points[i]-points[i-1]), np.square(points[i]-points[i-2]), np.square(points[i]-points[i+1]), np.square(points[i]-points[i+2])]
            #point_features = [np.square(points[i]-points[i-1]), np.square(points[i]-points[i+1])]
            pts[key + '_' + str(i)] = point_features

    return pts


if __name__ == "__main__":
    path = "/home/bella/Phd/code/code_bella/log/27/output/FIESTA/"
    pd_dice_per_slice = pd.read_csv(os.path.join(path,'dice_per_slice.csv'), index_col=0)
    pd_dice_per_slice.to_csv(os.path.join(path + 'test.csv'))
    dice_per_slice = pd_dice_per_slice.to_dict()

    middle_pts = middle_points(dice_per_slice)

    df_data = pd.DataFrame.from_dict(middle_pts).T
    model = IsolationForest(contamination=0.002)
    model.fit(df_data)

    predictions = model.predict(df_data)

    anomaly_indices = np.where(predictions==-1)
    anomaly_df = df_data.iloc[anomaly_indices]
    anomaly_df.to_csv(os.path.join(path + 'anomaly_detection/anomaly.csv'))
    df_data.to_csv(os.path.join(path, 'anomaly_detection/all.csv'))
    preds = predictions.tolist()
    print('stop')