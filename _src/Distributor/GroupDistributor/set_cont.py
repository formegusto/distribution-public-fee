import math as mt
import numpy as np


def set_cont(self):
    usages = self.meter_month['usage (kWh)'].values
    bins = round(mt.sqrt(usages.size / 2))

    y, x = np.histogram(usages, bins=bins)

    groups = np.zeros(usages.size)
    hist = list()
    for idx, _x in enumerate(x[:-1]):
        start = _x
        end = x[idx + 1]

        hist.append([
            start, end, (start + end) / 2
        ])
        groups[(usages >= start) & (usages <= end)] = idx
    hist = np.array(hist)

    # 재배치
    uni_groups = np.unique(groups).astype("int")
    hist = hist[uni_groups]
    new_groups = np.zeros(usages.size)
    for new_idx, idx in enumerate(uni_groups):
        new_groups[groups == idx] = new_idx

    group_cont = (hist / hist.sum(axis=0))[:, -1]
    cont_ = np.array([group_cont[int(_)] for _ in new_groups])
    self.cont_ = cont_ / cont_.sum()
