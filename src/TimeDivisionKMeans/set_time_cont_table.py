import numpy as np


def _time_cont_table(cont_table, division_size, size): return cont_table.reshape(
    size, -1, round(24 / division_size)).mean(axis=1).astype(np.float)


def set_time_cont_table(self):
    self.target_cont_ = _time_cont_table(
        self.cont_table_, self.division_size, len(self.df.columns))
    self.target_cluster_cont_ = _time_cont_table(
        self.cluster_cont_table_, self.division_size, self.K)
