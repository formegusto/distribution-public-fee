from sklearn.tree import DecisionTreeClassifier
import numpy as np


def _anomaly_update(x, anomaly_cols, new_labels):
    chk = np.where(anomaly_cols == x['가구명'])[0]

    if chk.size == 0:
        return x['label']
    else:
        return new_labels[chk[0]]


def adjust_anomaly_df(datas, anomaly, group_df):
    unanomaly_df = group_df[~np.isin(
        datas.columns, anomaly['가구명'])].copy()
    anomaly_df = anomaly.copy()

    X = unanomaly_df[['usage (kWh)']].values.copy()
    y = unanomaly_df[['label']].values

    dt = DecisionTreeClassifier()
    dt.fit(X, y)

    new_labels = dt.predict(anomaly_df[['usage (kWh)']].values)
    anomaly_cols = anomaly_df['가구명'].values

    group_df['label'] = group_df.apply(
        lambda x: _anomaly_update(x, anomaly_cols, new_labels), axis=1)

    _labels = group_df['label'].values
    new_labels = np.zeros(len(_labels)) - 1
    uni_labels = np.unique(_labels)

    for idx, _label in enumerate(uni_labels):
        new_labels[_labels == _label] = idx

    group_df['label'] = new_labels.astype("int")

    return group_df
