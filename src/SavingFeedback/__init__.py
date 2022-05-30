from .kmeans_run import kmeans_run
from .data_preprocessing import data_preprocessing
from .set_group import set_group
from .check_anomaly import check_anomaly
from .adjust_anomaly import adjust_anomaly
from .time_grouping import time_grouping
from .day_grouping import day_grouping
from ._feedback import time_feedback, day_feedback
from ._result import result
from .Drawing import Drawing
from .spread import spread

import numpy as np


class SavingFeedback:
    def __init__(self, xlsx, _type="kmeans", _tdtype="weight_mean", month=None):
        self._type = _type
        self._tdtype = _tdtype
        self.data_preprocessing(xlsx)

        if month is not None:
            self.select_month(month)

    def draw_init(self, name=None):
        draw = Drawing(self, name)
        draw.init_config()

        return draw

    def select_month(self, month):
        self.datas = self.m_60[self.m_60.index.month == month].copy()

    def time_based_grouping(self, time_size):
        self.mode = "time"
        self.time_size = time_size

        time_clusters = list()

        for c in self.kmeans.clusters_:
            time_group = self.time_grouping(c, time_size)
            # 0 : cluster pattern, 1: mean cluster pattern
            time_clusters.append(time_group)

        # Recovery Group Index
        self.group_index = self.time_grouping(
            self.datas.index.values, time_size, need_mean=False
        ).flatten()
        self.clusters_ = time_clusters

    def day_based_grouping(self):
        self.mode = "day"
        day_clusters = list()

        for c in self.kmeans.clusters_:
            day_group = self.day_grouping(c)
            # 0 : cluster pattern, 1: mean cluster pattern
            day_clusters.append(day_group)

        # Recovery Group Index
        _group_index = self.day_grouping(
            self.datas.index.values, need_mean=False)

        # Flatten
        group_index = np.array([], dtype=np.datetime64)
        for _idx in _group_index:
            group_index = np.append(group_index, _idx)
        self.group_index = group_index
        self.clusters_ = day_clusters

    def feedback(self):
        simulations = list()
        feedback_func = self.time_feedback if self.mode == "time" else self.day_feedback
        for name in self.group['name']:
            simulations.append(feedback_func(name))

        self.simulations = simulations

    def recovery(self):
        recoveries = np.array([])
        for sim in self.simulations:
            spread_pat = self.spread(sim)
            _recovery = spread_pat[self.group_index.argsort()]

            recoveries = np.append(recoveries, _recovery)

        self.recoveries = recoveries.reshape(-1, len(self.datas))


SavingFeedback.data_preprocessing = data_preprocessing
SavingFeedback.kmeans_run = kmeans_run
SavingFeedback.set_group = set_group
SavingFeedback.check_anomaly = check_anomaly
SavingFeedback.adjust_anomaly = adjust_anomaly
SavingFeedback.time_grouping = time_grouping
SavingFeedback.day_grouping = day_grouping
SavingFeedback.time_feedback = time_feedback
SavingFeedback.day_feedback = day_feedback
SavingFeedback.result = result
SavingFeedback.spread = spread
