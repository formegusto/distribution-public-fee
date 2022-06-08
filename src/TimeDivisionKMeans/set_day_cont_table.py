import numpy as np


def _day_cont_table(cont_table, idx, size):
    day_cont_table = None

    for weekday in range(0, 7):
        _day_cont_table = cont_table[:, idx.weekday == weekday].mean(
            axis=1).reshape(size, 1)
        if weekday == 0:
            day_cont_table = _day_cont_table
        else:
            day_cont_table = np.append(day_cont_table, _day_cont_table, axis=1)

    return day_cont_table


def set_day_cont_table(self):
    self.target_cont_ = _day_cont_table(
        self.cont_table_, self.idx, self.df.columns.size)
    self.target_cluster_cont_ = _day_cont_table(
        self.cluster_cont_table_, self.idx, self.K)
