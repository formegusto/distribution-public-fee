import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Draw Contribution Plot


def draw_cont_plot(self, col_size=3):
    matplotlib.rc('font', family='AppleGothic')
    plt.rcParams['axes.unicode_minus'] = False
    r, c = np.arange((round((self.K - 1) / col_size) + 1)
                     * col_size).reshape(-1, col_size).shape
    plt.figure(figsize=(16, 5*r))

    for _label in range(r * c):
        if _label >= self.K:
            break

        cluster_cont = self.cluster_cont_table_[_label]
        ax = plt.subplot(r, c, _label + 1)
        ax.plot(cluster_cont, color='g', linewidth=2)
        ax.plot(self.cont_table_[self.labels_ ==
                _label].T, color='g', linewidth=0.1)

        ax.text(0.98, 0.925, "Contribution : {}".format(
                round(cluster_cont.mean())),
                ha="right",
                va="center",
                transform=ax.transAxes,
                fontsize=16)
        ax.set_title("클러스터 {}".format(_label))

    plt.show()
