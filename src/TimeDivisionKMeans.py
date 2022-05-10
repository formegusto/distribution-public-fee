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

        groups_ = np.array([])

        for col in cluster_info.columns:
            _groups = cluster_info.groupby(col).count()
            max_group = _groups.index[_groups.values[:, 0].argmax()]

            groups_ = np.append(groups_, max_group)

        self.groups_ = groups_
