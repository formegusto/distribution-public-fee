import numpy as np
import pandas as pd


def get_anomaly_df(group_df):
    min_anomaly_data = pd.DataFrame()

    for label in np.unique(group_df['label']):
        now_step_df = group_df[group_df['label'] == label]
        next_step_min = group_df[group_df['label']
                                 == (label + 1)]['usage (kWh)'].min()
        chk_idx = now_step_df['usage (kWh)'] >= next_step_min\

        chk = now_step_df[chk_idx]

        if len(chk) != 0:
            min_anomaly_data = min_anomaly_data.append(
                chk
            )

    return min_anomaly_data
