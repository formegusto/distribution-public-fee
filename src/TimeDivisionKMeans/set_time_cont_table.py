import numpy as np


def _time_cont_table(cont_table, division_size, size): return cont_table.reshape(
    size, -1, round(24 / division_size)).mean(axis=1).astype(np.float)


def _weight_time_cont_table(cont_table, division_size, size, weights):
    cont_weights = (cont_table * weights)
    cont_weights = cont_weights.reshape(size, -1, round(24 / division_size))
    time_cont = (cont_weights.sum(axis=1) / weights.reshape(-1,
                                                            round(24 / division_size)).sum(axis=0)).astype(np.float).round()

    return time_cont


def set_time_cont_table(self):
    self.target_cont_ = _time_cont_table(
        self.cont_table_, self.division_size, len(self.df.columns))
    self.target_cluster_cont_ = _time_cont_table(
        self.cluster_cont_table_, self.division_size, self.K)
    # self.target_cont_ = _weight_time_cont_table(
    #     self.cont_table_, self.division_size, len(self.df.columns), self.weights_)
    # self.target_cluster_cont_ = _weight_time_cont_table(
    #     self.cluster_cont_table_, self.division_size, self.K, self.weights_)
