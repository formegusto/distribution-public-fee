import numpy as np


def time_feedback(self, name):
    target_house = self.group[self.group['name'] == name]
    target_pattern = self.datas[name].values
    # print("\n절약 전 사용량", round(target_pattern.sum()))
    time_group, mean_time_group = self.time_grouping(
        target_pattern, self.time_size)

    label = target_house['label'].values[0]
    if label == 0:
        return time_group

    _now = (self.clusters_[label][1].mean(axis=1) *
            1000).astype(np.float).round() / 1000
    _prev = (self.clusters_[label - 1][1].mean(axis=1)
             * 1000).astype(np.float).round() / 1000
    _target = (mean_time_group.mean(axis=1) *
               1000).astype(np.float).round() / 1000

    if self._type == "tdkmeans":
        _now_cont = self.kmeans.target_cluster_cont_[label]
        _prev_cont = self.kmeans.target_cluster_cont_[label - 1]
        _target_cont = self.kmeans.target_cont_[
            self.datas.columns == name
        ][0]
        now_saving_point = np.where(_target_cont > _now_cont + 1)[0]
        prev_saving_point = np.where(_target_cont > _prev_cont + 1)[0]
        prev_saving_point = prev_saving_point[~np.isin(
            prev_saving_point, now_saving_point)]

    else:
        now_saving_point = np.where(_target > _now)[0]
        prev_saving_point = np.where(_target > _prev)[0]
        prev_saving_point = prev_saving_point[~np.isin(
            prev_saving_point, now_saving_point)]

    err = np.zeros(len(time_group))
    err[now_saving_point] = _target[now_saving_point] - _now[now_saving_point]
    err[prev_saving_point] = _target[prev_saving_point] - _prev[prev_saving_point]

    if np.any(err < 0):
        print(_target[now_saving_point], _now[now_saving_point])
        print(_target[prev_saving_point], _prev[prev_saving_point])

        print(_now_cont, _prev_cont, _target_cont)
        print(name, err)

    sims = time_group.copy()
    neg_mem = list()

    for idx, sim in enumerate(sims):
        _err = err[idx]

        if _err == 0:
            continue
        sim -= err[idx]

        neg_err = sim[sim < 0]
        sim[sim < 0] = 0

        for neg in neg_err:
            chk = sim[sim > abs(neg)]
            if chk.size != 0:
                sim[sim > abs(neg)][0] += neg
            else:
                neg_mem.append(neg)

    # print("모든 사용량이 피드백 되었나요?", sum(neg_mem) <= 0)
    # print("실천 최대 기대값 {}kWh \n".format(round(sims.sum())))

    return sims


def day_feedback(self, name):
    target_house = self.group[self.group['name'] == name]
    target_pattern = self.datas[name].values
    # print("\n절약 전 사용량 {}kWh".format(round(target_pattern.sum())))
    time_group, mean_time_group = self.day_grouping(target_pattern)

    label = target_house['label'].values[0]
    if label == 0:
        return time_group

    _now = (self.clusters_[label][1].mean(axis=1) *
            1000).astype(np.float).round() / 1000
    _prev = (self.clusters_[label - 1][1].mean(axis=1)
             * 1000).astype(np.float).round() / 1000
    _target = (mean_time_group.mean(axis=1) *
               1000).astype(np.float).round() / 1000

    now_saving_point = np.where(_target > _now)[0]
    prev_saving_point = np.where(_target > _prev)[0]
    prev_saving_point = prev_saving_point[~np.isin(
        prev_saving_point, now_saving_point)]

    err = np.zeros(len(time_group))
    err[now_saving_point] = _target[now_saving_point] - _now[now_saving_point]
    err[prev_saving_point] = _target[prev_saving_point] - _prev[prev_saving_point]

    sims = time_group.copy()
    neg_mem = list()

    for idx, sim in enumerate(sims):
        sim -= err[idx]

        neg_err = sim[sim < 0]
        sim[sim < 0] = 0

        for neg in neg_err:
            chk = sim[sim > abs(neg)]
            if chk.size != 0:
                sim[sim > abs(neg)][0] += neg
            else:
                neg_mem.append(neg)
    # print("모든 사용량이 피드백 되었나요?", sum(neg_mem) <= 0)
    # print("실천 최대 기대값 {}kWh \n".format(
    #     round(sum([_.sum() for _ in sims]))))

    return sims
