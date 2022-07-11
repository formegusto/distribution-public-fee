import matplotlib
import matplotlib.pyplot as plt
from ...Calculator.common import STEP_LIMITS_HOUSEHOLD


def col_monitor(self, public_mean_kwh=0):
    matplotlib.rc('font', family='AppleGothic')
    plt.rcParams['axes.unicode_minus'] = False

    calc = self.analyzer.calc
    household_count = calc.household_count
    household_kwh = calc.household_kwh
    households_kwh = calc.datas['usage (kWh)'].values
    apt = household_kwh + (public_mean_kwh * household_count)
    apt_mean = round(apt / household_count)
    self.analyzer.set_calc(kwh=apt)

    plt.figure(figsize=(18, 4))
    STEP_LIMIT = STEP_LIMITS_HOUSEHOLD[calc.season].copy()
    LIMIT_COLOR = ['yellow', 'orange']
    for idx in range(1, len(STEP_LIMIT)):
        STEP_LIMIT[idx] = STEP_LIMIT[idx - 1] + STEP_LIMIT[idx]

    titles = ['수거 현황', '잉여 사용량', '초과수거 사용량']
    x_range = range(0, household_count)
    y_values = [
        [(apt_mean, 'r'), (households_kwh, 'b')],
        [(apt_mean, 'r'), (households_kwh, 'white')],
        [(households_kwh, 'b'), (apt_mean, 'white')]
    ]
    for idx, title in enumerate(titles):
        ax = plt.subplot(1, 3, idx + 1)
        y_value = y_values[idx]
        for y, c in y_value:
            ax.bar(x_range, y, color=c)
        for c_idx, limit in enumerate(STEP_LIMIT):
            c = LIMIT_COLOR[c_idx]
            ax.axhline(limit, color=c)

        ax.set_title(title)

    plt.show()
