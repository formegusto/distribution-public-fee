import inspect
import math as mt
import matplotlib
import matplotlib.pyplot as plt


def _drawing(self, pat):
    if self.sv.mode == "time":
        titles = ["{}시 ~ {}시".format(start_time, start_time + (self.sv.time_size-1))
                  for start_time in range(0, 24, self.sv.time_size)]
        xticks = [range(start_time, start_time + self.sv.time_size)
                  for start_time in range(0, 24, self.sv.time_size)]
    elif self.sv.mode == "day":
        titles = ['월', '화', '수', '목', '금', '토', '일']
        xticks = [range(0, 24, 5) for _ in range(7)]

    matplotlib.rc('font', family='AppleGothic')
    plt.rcParams['axes.unicode_minus'] = False

    plt.figure(figsize=(20, (mt.floor((len(pat) - 1) / 3) + 1) * 6))

    for idx, group in enumerate(pat):
        _marker = self.marker[idx]
        mean_group = group.mean(axis=0)
        _my_mean = round(mean_group.mean() * 1000) / 1000
        ax = plt.subplot((mt.floor((len(pat) - 1) / 3) + 1), 3, idx+1)

        color = 'g'
        need_saving = None
        if inspect.stack()[1][3] == "house":
            now_mean = (self.now_pattern[idx].mean(
                axis=0) * 1000).astype("float").round() / 1000

            if _marker != 0:
                if _marker == 1:
                    color = 'r'
                    need_saving = round(
                        (_my_mean - now_mean.mean()) * 1000) / 1000
                else:

                    prev_mean = (self.prev_pattern[idx].mean(
                        axis=0) * 1000).astype("float").round() / 1000
                    color = 'darkorange'
                    need_saving = round(
                        (_my_mean - prev_mean.mean()) * 1000) / 1000

        ax.plot(group.T, color=color, linewidth=0.3)
        ax.plot(mean_group, color=color, linewidth=1)
        ax.set_title(titles[idx])
        ax.text(0.95, 0.95,
                "{} kWh".format(_my_mean),
                fontsize=16,
                ha="right",
                va="top",
                transform=ax.transAxes)
        if need_saving is not None:
            ax.text(0.05, 0.05,
                    "절약 필요 사용량 : {} kWh".format(need_saving),
                    fontsize=12,
                    ha="left",
                    va="bottom",
                    fontweight="bold",
                    transform=ax.transAxes)
        plt.xticks(xticks[idx])
        plt.xlabel("시간")
        plt.ylabel("사용량 (kWh)")

    plt.show()


class Drawing:
    def __init__(self, saving_feedback, name=None):
        if name is None:
            self.random = True
        else:
            self.name = name
            self.random = False

        self.sv = saving_feedback

    def init_config(self):
        if self.random:
            target_house = self.sv.group.sample(n=1).copy()
        else:
            target_house = self.sv.group[self.sv.group['name'] == self.name].copy(
            )
        self.target_house = target_house

        self.name = target_house['name'].values[0]

        self.label = target_house['label'].values[0]
        target_pattern = self.sv.datas[self.name].values
        self.marker = self.sv.markers[
            self.sv.group[self.sv.group['name'] == self.name].index[0]
        ]

        mode = self.sv.mode
        if mode == "time":
            self.house_pattern = self.sv.time_grouping(
                target_pattern, self.sv.time_size)[0]
        elif mode == "day":
            self.house_pattern = self.sv.day_grouping(target_pattern)[0]

        self.now_pattern = self.sv.clusters_[self.label][0]
        if self.label != 0:
            self.prev_pattern = self.sv.clusters_[self.label - 1][0]

    def house(self):
        print("가구 명 : {}".format(self.name))
        self._drawing(self.house_pattern)

    def now(self):
        print("현재 가구가 속해 있는 기여도 {}번 그룹".format(self.label))
        self._drawing(self.now_pattern)

    def prev(self):
        if self.label == 0:
            print("최소 사용량 기여도 그룹의 가구 입니다.")
        else:
            print("현재 가구가 속해 있는 기여도 그룹의 이전 그룹".format(self.label - 1))
            self._drawing(self.prev_pattern)

    def feedback(self):
        feedback_func = self.sv.time_feedback if self.sv.mode == "time" else self.sv.day_feedback
        sims = feedback_func(self.name)

        self._drawing(sims)


Drawing._drawing = _drawing
