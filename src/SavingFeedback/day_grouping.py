import numpy as np


def day_grouping(self, target):
    day_group = list()
    mean_day_group = list()

    for idx in range(7):
        _day_group = target[
            self.datas.index.weekday == idx].reshape(-1, 24)
        day_group.append(_day_group)
        mean_day_group.append(_day_group.mean(axis=0))

    return day_group, mean_day_group
