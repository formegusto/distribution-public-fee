import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def get_percentage(self,
                   start_percentage=10,
                   end_percentage=50, inc=1,
                   var_point=False):
    matplotlib.rc('font', family='AppleGothic')
    plt.rcParams['axes.unicode_minus'] = False

    anal_range = range(start_percentage, end_percentage, inc)
    mean_apts = np.array([])
    result = None
    for public_percentage in anal_range:
        self.analyzer.set_bill_table(public_percentage=public_percentage)
        mean_apts = np.append(mean_apts, self.analyzer.calc.apt_obj.kwh)
        bill_table = self.analyzer.bill_table.get

        basic_rates = bill_table[['기본요금']].to_numpy()[:3]
        if result is None:
            result = basic_rates
        else:
            result = np.append(result, basic_rates, axis=1)

    plt.figure(figsize=(16, 8))
    indexes = ['APT 기본요금', '세대부 기본요금', '공용부 기본요금']
    palette = plt.cm.get_cmap("Set1", result[:, 0].size)
    for idx, index_name in enumerate(indexes):
        if var_point:
            plt.plot(anal_range, result[idx], c=palette(
                idx), label=index_name, alpha=0.5)
        else:
            plt.plot(anal_range, result[idx],
                     c=palette(idx), label=index_name)
    cols = np.array([_ for _ in anal_range])

    if var_point:
        y, x = np.unique(result.T, axis=0, return_index=True)
        x = cols[x]
        for idx, _x in enumerate(x):
            _y = y[idx]
            _x = np.full(_y.size, _x)
            plt.scatter(_x, _y, s=200, color=palette(idx))

        result = np.append(
            result, mean_apts.reshape(-1, mean_apts.size), axis=0)
        indexes += ['APT 평균사용량']

    plt.title("공용부 사용량 증가에 따른 기본요금 변화")
    plt.ylabel("요금")
    plt.xlabel("공용부 percentage")
    plt.legend(loc='upper left')
    plt.show()

    return pd.DataFrame(result, columns=["{}%".format(_) for _ in cols], index=indexes)
