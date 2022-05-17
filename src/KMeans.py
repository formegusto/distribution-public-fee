import math as mt
from colorama import init
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances as euc


def setting_ran_K(K, datas, mean_init=False):
    data_idxes = np.arange(len(datas))

    return datas[np.random.choice(data_idxes, K)]


def setting_init_K(K, datas, mean_init=False):
    r, c = datas.shape

    clusters_ = np.array([])
    k_setting_datas = datas.copy()

    if mean_init:
        mean_pat = np.expand_dims(datas.mean(axis=0), axis=0)
        next_K = euc(mean_pat, datas).argmin()
    else:
        data_idxes = np.arange(len(datas))
        next_K = np.random.choice(data_idxes)
        del data_idxes

    while True:
        clusters_ = np.append(clusters_, k_setting_datas[next_K])
        clusters_ = clusters_.reshape(-1, c)

        if K == len(clusters_):
            break
        else:
            k_setting_datas = np.delete(k_setting_datas, next_K, axis=0)
            dis_check = euc(clusters_, k_setting_datas).mean(axis=0)

            next_K = dis_check.argmax()

    del next_K
    del k_setting_datas

    return clusters_

# 정석 kmeans++


def setting_init_K_ver_2(K, datas, mean_init=False):
    r, c = datas.shape

    # 평균에서 가장 가까운 친구
    if mean_init:
        mean_pat = np.expand_dims(datas.mean(axis=0), axis=0)
        min_idx = euc(mean_pat, datas).argmin()
        clusters_ = np.expand_dims(datas[min_idx], axis=0)
    else:
        clusters_ = np.expand_dims(
            datas[np.random.randint(r)], axis=0)

    for c_id in range(K - 1):
        dist = euc(datas, clusters_).min(axis=1)
        next_centroid = datas[np.argmax(dist), :]

        clusters_ = np.append(clusters_, [next_centroid], axis=0)

    return clusters_


@property
def wss(self):
    _uni_labels = np.unique(self.labels_).astype("int")
    wss = 0

    for _label in _uni_labels:
        _cluster = self.clusters_[_label].reshape(1, -1)
        _data = self.datas[self.labels_ == _label]

        wss += (euc(_cluster, _data)[0] ** 2).sum()

    return wss


@property
def ecv(self):
    return 1 - (self.wss / self.tss)


init_centrois_func = [setting_ran_K, setting_init_K, setting_init_K_ver_2]


class KMeans():
    def __init__(self, datas, ver=2, K=None):
        self.datas = datas
        self.memory = []
        self.ecv_memory = np.array([])
        self.ver = ver
        if K is None:
            self.K = round(mt.sqrt(len(self.datas) / 2))
        else:
            self.K = K

    def init_setting(self, mean_init=False):
        init_func = init_centrois_func[self.ver -
                                       1]
        self.clusters_ = init_func(self.K, self.datas, mean_init)

        _mean = self.datas.mean(axis=0)
        self.mean = _mean
        _mean = _mean.reshape(-1, len(_mean))
        self.tss = (euc(_mean, self.datas) ** 2).sum()
        self.labels_ = np.zeros(self.K)

    def next_setting(self):
        _clusters = self.clusters_.copy()
        _datas = self.datas.copy()
        _labels = self.labels_.copy()

        _uni_labels = np.unique(_labels)

        for _label in _uni_labels:
            _data = _datas[_labels == _label]
            _clusters[_label] = _data.mean(axis=0)

        self.clusters_ = _clusters

        del _clusters
        del _datas
        del _labels
        del _uni_labels

    def next(self):
        _clusters = self.clusters_.copy()
        _datas = self.datas.copy()

        dis_chk = euc(_datas, _clusters)
        _labels = dis_chk.argmin(axis=1)

        self.labels_ = _labels

        self.next_setting()

        del _clusters
        del _datas

    def fit(self, early_stop_cnt=3, memory=True, mean_init=False, logging=True):
        self.init_setting(mean_init)
        _early_stop_cnt = 0
        while True:
            bak_labels = self.labels_.copy()
            self.next()
            _labels = self.labels_.copy()

            if np.array_equiv(bak_labels, _labels):
                _early_stop_cnt += 1

            if _early_stop_cnt >= early_stop_cnt:
                if memory:
                    self.memory.append(
                        [self.clusters_.copy(), self.labels_.copy()])
                    self.ecv_memory = np.append(
                        self.ecv_memory, self.ecv)
                if logging:
                    print("ECV : {} %".format(round(self.ecv * 100)))
                break

    def draw_plot(self, col_size=3):
        matplotlib.rc('font', family='AppleGothic')
        plt.rcParams['axes.unicode_minus'] = False

        _labels = np.unique(self.labels_)
        r, c = np.arange((round((self.K - 1) / col_size) + 1)
                         * col_size).reshape(-1, col_size).shape
        plt.figure(figsize=(16, 5*r))

        for _label in range(r * c):
            ax = plt.subplot(r, c, _label + 1)

            if len(_labels) <= _label:
                ax.set_visible(False)
            else:
                plt.plot(self.datas[self.labels_ == _label].T,
                         c='g', linewidth=0.3)
                plt.plot(self.clusters_[_label], c='b', linewidth=0.5)
                plt.yticks(np.arange(0, self.datas.max(), 0.5))
                ax.text(0.98, 0.925, "{} kWh".format(
                    round(self.clusters_[_label].sum())),
                    ha="right",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=16)
                ax.set_title("클러스터 {}".format(_label))

        plt.show()

    def draw_all_plot(self):
        matplotlib.rc('font', family='AppleGothic')
        plt.rcParams['axes.unicode_minus'] = False

        plt.figure(figsize=(16, 4))

        plt.plot(self.datas[0], color='g', linewidth=0.1, label='데이터')
        plt.plot(self.clusters_[0], color='b', linewidth=0.5, label='클러스터')

        plt.plot(self.datas[1:].T, color='g', linewidth=0.1)
        plt.plot(self.clusters_[1:].T, color='b', linewidth=0.5)

        plt.title("클러스터 현황")
        plt.legend()

        plt.show()

    def sorting(self):
        clusters = self.clusters_.copy()
        labels = self.labels_.copy()

        clusters_total = clusters.sum(axis=1)

        sort_idxes = clusters_total.argsort()
        sort_clusters = clusters[sort_idxes].copy()

        sort_labels = np.zeros(len(labels)) - 1

        for label in range(len(clusters)):
            sort_labels[labels == sort_idxes[label]] = label

        self.clusters_ = sort_clusters
        self.labels_ = sort_labels


KMeans.wss = wss
KMeans.ecv = ecv
