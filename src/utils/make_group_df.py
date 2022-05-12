import numpy as np
import pandas as pd


def make_group_df(datas, kmeans, type="kmeans"):
    kmeans.sorting()

    group_df = pd.DataFrame(
        np.column_stack(
            [datas.columns.values, datas.sum(axis=0).round().astype("int")]),
        columns=['가구명', 'usage (kWh)']
    )
    group_df['label'] = kmeans.labels_.astype("int")

    return group_df
