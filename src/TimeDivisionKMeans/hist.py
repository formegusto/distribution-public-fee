import datetime as dt
import numpy as np

import matplotlib
import matplotlib.pyplot as plt


def hist(self, division_slot=0):

    plt.figure(figsize=(16, 4))
    kmeans_idx = np.arange(self.division_round)[
        self.idx.time == dt.time(self.division_size * division_slot)]

    for idx in kmeans_idx:
        kmeans = self.kmeans_[idx]
        uni_labels = np.unique(kmeans.labels_)
        colors = plt.cm.get_cmap("YlGn", uni_labels.size)
        sums = kmeans.datas.sum(axis=1)
        for label in uni_labels:
            plt.hist(sums[kmeans.labels_ == label],
                     label="클러스터 {}".format(int(label)), color=colors(label),
                     alpha=0.5)

    plt.xlabel("Usage (kWh)")
    plt.ylabel("Number of Samples")
    plt.show()
