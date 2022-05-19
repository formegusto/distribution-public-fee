import numpy as np
import pandas as pd


def make_group_df(datas, kmeans, _type="kmeans"):
    if _type == "kmeans":
        kmeans.sorting()

    group_df = pd.DataFrame(
        np.column_stack(
            [datas.columns.values, datas.sum(axis=0).astype(np.float).round().astype("int")]),
        columns=['가구명', 'usage (kWh)']
    )
    if _type == "kmeans":
        group_df['label'] = kmeans.labels_.astype("int")
    else:
        group_df['label'] = kmeans.groups_.astype("int")

    return group_df
