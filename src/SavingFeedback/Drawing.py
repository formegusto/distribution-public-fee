import math as mt
import matplotlib
import matplotlib.pyplot as plt


def time_drawing(self, pat):
    if self.sv.mode == "time":
        titles = ["{}시 ~ {}시".format(start_time, start_time + (self.sv.time_size-1))
                  for start_time in range(0, 24, self.sv.time_size)]
        xticks = [range(start_time, start_time + self.sv.time_size)
                  for start_time in range(0, 24, self.sv.time_size)]
    elif self.sv.mode == "day":
        titles = ['월', '화', '수', '목', '금', '토', '일']
        xticks = [range(24) for _ in range(7)]

    matplotlib.rc('font', family='AppleGothic')
    plt.rcParams['axes.unicode_minus'] = False

    plt.figure(figsize=(20, (mt.floor((len(pat) - 1) / 3) + 1) * 6))

    for idx, group in enumerate(pat):
        mean_group = group.mean(axis=0)
        ax = plt.subplot((mt.floor((len(pat) - 1) / 3) + 1), 3, idx+1)

        ax.plot(xticks[idx], group.T, color='g', linewidth=0.25)
        ax.plot(xticks[idx], mean_group, color='g', linewidth=1)
        ax.set_title(titles[idx])
        ax.text(0.95, 0.95,
                "{} kWh".format(round(mean_group.mean() * 1000) / 1000),
                fontsize=16,
                ha="right",
                va="top",
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
        self.time_drawing(self.house_pattern)

    def now(self):
        print("현재 가구가 속해 있는 기여도 {}번 그룹".format(self.label))
        self.time_drawing(self.now_pattern)

    def prev(self):
        if self.label == 0:
            print("최소 사용량 기여도 그룹의 가구 입니다.")
        else:
            print("현재 가구가 속해 있는 기여도 그룹의 이전 그룹".format(self.label - 1))
            self.time_drawing(self.prev_pattern)

    def feedback(self):
        pass


Drawing.time_drawing = time_drawing
