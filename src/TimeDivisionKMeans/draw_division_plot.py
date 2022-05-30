import matplotlib
import matplotlib.pyplot as plt


def draw_division_plot(self, division_round):
    matplotlib.rc('font', family='AppleGothic')
    plt.rcParams['axes.unicode_minus'] = False

    plt.figure(figsize=(3, 8))

    start_time = (division_round * 3) % 24
    end_time = ((division_round + 1) * 3) % 24

    if end_time == 0:
        end_time = 24

    plt.plot(range(start_time, end_time, 1),
             self.datas[:, division_round, :].T, color='g', linewidth=0.25)
    plt.xticks(range(start_time, end_time, 1))
    plt.xlabel("시간 (hours)")
    plt.ylabel("사용량 (kWh)")
    plt.title(self.df.index[division_round * 3].strftime("%Y-%m-%d"))

    plt.show()
