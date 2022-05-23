from .kmeans_run import kmeans_run
from .data_preprocessing import data_preprocessing
from .set_group import set_group
from .check_anomaly import check_anomaly
from .adjust_anomaly import adjust_anomaly
from .time_grouping import time_grouping


class SavingFeedback:
    def __init__(self, xlsx, month=None):
        self.data_preprocessing(xlsx)

        if month is not None:
            self.select_month(month)

    def select_month(self, month):
        self.datas = self.m_60[self.m_60.index.month == 1].copy()

    def time_based_grouping(self, time_size):
        time_clusters = list()
        for c in self.kmeans.clusters_:
            time_group = self.time_grouping(c, time_size)
            # 0 : cluster pattern, 1: mean cluster pattern
            time_clusters.append(time_group)

        self.clusters_ = time_clusters


SavingFeedback.data_preprocessing = data_preprocessing
SavingFeedback.kmeans_run = kmeans_run
SavingFeedback.set_group = set_group
SavingFeedback.check_anomaly = check_anomaly
SavingFeedback.adjust_anomaly = adjust_anomaly
SavingFeedback.time_grouping = time_grouping
