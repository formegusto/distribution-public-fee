import pandas as pd
import numpy as np
from ..KMeans import KMeans
from .draw_division_plot import draw_division_plot
from .draw_cont_plot import draw_cont_plot
from .set_time_cont_table import set_time_cont_table
from .set_day_cont_table import set_day_cont_table


class TimeDivisionKMeans:
    def __init__(self, datas, division_size=3):
        if (24 % division_size) != 0:
            raise ValueError("only 24 % division_size == 0 !!")

        self.df = datas.copy()
        self.datas = datas.T.values.reshape(-1, round(len(datas) /
                                                      division_size), division_size)
        self.idx = datas.index[::division_size]
        self.division_size = division_size
        c = len(datas)
        self.division_round = round(c / division_size)

    def fit(self, _type="weight_mean"):
        self.kmeans_ = list()

        for _round in range(self.division_round):
            kmeans = KMeans(datas=self.datas[:, _round])
            kmeans.fit(logging=False)
            kmeans.sorting()
            kmeans.adjust_anomaly()

            self.kmeans_.append(kmeans)
            if (_round % 10 == 0) or \
                    (_round + 1 == self.division_round):
                print("{}/{} - ECV:{}%".format(_round+1,
                      self.division_round, round(kmeans.ecv * 100)))

        cluster_info = pd.DataFrame(columns=self.df.columns)
        cluster_info.index = pd.Series(name="division_round")

        for _round, _kmeans in enumerate(self.kmeans_):
            cluster_info.loc[_round] = _kmeans.labels_.astype("int")

        self.cluster_info = cluster_info.copy()
        self.cont_table_ = cluster_info.copy().to_numpy().T

        # 방법 1
        # labels_ = np.array([])
        # for col in self.cluster_info.columns:
        #     # counts = self.cluster_info[col].value_counts()
        #     # max_group = ((counts.index + 1) * counts).idxmax()
        #     max_group = self.cluster_info[col].value_counts().idxmax()

        #     labels_ = np.append(labels_, max_group)
        # self.labels_ = labels_.astype("int")

        # 방법 1. mean
        if _type == "mean":
            labels_ = self.cluster_info.mean().round().astype("int").values
        elif _type == "weight_mean":
            # 방법 2. weight_mean
            labels_ = np.array([])

            sum_datas = self.datas.sum(axis=2).sum(axis=0)
            sum_total = self.datas.sum(axis=2).sum(axis=0).sum()

            weights = sum_datas / sum_total
            self.weights_ = weights

            for col in self.cluster_info.columns:
                max_group = round(
                    (self.cluster_info[col] * weights).sum() / weights.sum())

                labels_ = np.append(labels_, max_group)

        labels_ = labels_.astype("int")
        uni_labels = np.unique(labels_)
        self.labels_ = np.zeros(len(labels_)) - 1
        for idx, _label in enumerate(uni_labels):
            self.labels_[labels_ == _label] = idx

        self.labels_ = self.labels_.astype("int")

        clusters_ = np.array([])
        _pats = self.df.to_numpy().T

        for label in np.unique(self.labels_):
            mean_pat = _pats[self.labels_ == label].mean(axis=0)
            clusters_ = np.append(clusters_, mean_pat)
            clusters_ = clusters_.reshape(-1, mean_pat.size)

        self.clusters_ = clusters_

        kmeans_ = KMeans(datas=self.df.T.values, K=len(clusters_))
        kmeans_.labels_ = labels_.astype("int")
        kmeans_.clusters_ = clusters_

        self.kmeans = kmeans_

        self.set_cluster_cont_table()

    def set_cluster_cont_table(self):
        unique_labels = np.unique(self.labels_)
        cluster_cont_table = np.array([])

        for label in unique_labels:
            cols = self.cluster_info.columns[self.labels_ == label]
            cluster_cont_table = np.append(cluster_cont_table,
                                           self.cluster_info[cols].mean(axis=1).round().astype("int").values)

        self.cluster_cont_table_ = cluster_cont_table.reshape(
            -1, self.division_round)

    def draw_plot(self):
        self.kmeans.draw_plot()

    def next_setting(self):
        self.kmeans.next_setting()

        self.lables_ = self.kmeans.labels_
        self.clusters_ = self.kmeans.clusters_


TimeDivisionKMeans.set_time_cont_table = set_time_cont_table
TimeDivisionKMeans.set_day_cont_table = set_day_cont_table
TimeDivisionKMeans.draw_cont_plot = draw_cont_plot
TimeDivisionKMeans.draw_division_plot = draw_division_plot
