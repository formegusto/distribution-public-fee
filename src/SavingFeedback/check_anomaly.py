import numpy as np
import pandas as pd


def check_anomaly(self):
    uni_labels = np.unique(self.group['label'].values)
    anomaly = pd.DataFrame()

    for idx, _label in enumerate(uni_labels[:-1]):
        _now = self.group[self.group['label'] == _label]
        _next = self.group[self.group['label'] == (_label + 1)]
        _chk = (_now['usage (kWh)'] > _next['usage (kWh)'].min()).values

        _anomaly = _now[_chk]
        if len(_anomaly) != 0:
            anomaly = anomaly.append(_anomaly, ignore_index=True)

    return anomaly
