import pandas as pd
import numpy as np
from .KMeans import KMeans


class TimeDivisionKMeans:
    def __init__(self, datas, division_size=3):
        if (24 % division_size) != 0:
            raise ValueError("only 24 % division_size == 0 !!")

        self.df = datas.copy()
        self.datas = datas.T.values.reshape(-1, round(len(datas) /
                                                      division_size), division_size)
        c = len(datas)
        self.division_round = round(c / division_size)

    def fit(self):
        self.kmeans_ = list()

        for _round in range(self.division_round):
            kmeans = KMeans(datas=self.datas[:, _round])
            kmeans.fit(logging=False)
            kmeans.sorting()

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

        labels_ = np.array([])
        for col in self.cluster_info.columns:
            max_group = self.cluster_info[col].value_counts().idxmax()

            labels_ = np.append(labels_, max_group)

        self.labels_ = labels_.astype("int")

        _unique_labels = np.unique(labels_)
        groups_ = np.zeros(len(labels_)) - 1

        for idx, label in enumerate(_unique_labels):
            groups_[labels_ == label] = idx

        self.groups_ = groups_.astype("int")
