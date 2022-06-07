import numpy as np


def set_time_cont_table(self):
    self.target_cont_ = self.cont_table_.reshape(
        len(self.df.columns), -1, round(24 / self.division_size)).mean(axis=1).astype(np.float)

    self.target_cluster_cont_ = self.cluster_cont_table_.reshape(
        self.K, -1, round(24 / self.division_size)).mean(axis=1).astype(np.float)
