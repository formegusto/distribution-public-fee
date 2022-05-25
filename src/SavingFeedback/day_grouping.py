import numpy as np


def day_grouping(self, target, need_mean=True):
    day_group = list()
    mean_day_group = list()

    for idx in range(7):
        _day_group = target[
            self.datas.index.weekday == idx].reshape(-1, 24)
        day_group.append(_day_group)
        if need_mean:
            mean_day_group.append(_day_group.mean(axis=0))

    if need_mean:
        return day_group, np.array(mean_day_group)
    else:
        return day_group
