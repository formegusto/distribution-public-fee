import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from ..Calculator import Calculator


def get_col_burden(self):
    matplotlib.rc('font', family='AppleGothic')
    plt.rcParams['axes.unicode_minus'] = False

    basic_col_burden = np.array(
        [self.calc.apt_obj.basic / _.basic for _ in self.calc.households])

    elec_rate_col_burden = list()
    _datas = self.calc.datas.copy()
    public_kwh = self.calc.public_kwh
    pri_public_won = round(self.calc.public_won / self.calc.household_count)

    for idx, target_data in self.calc.datas.iterrows():
        _datas['usage (kWh)'] = target_data['usage (kWh)']
        apt = _datas['usage (kWh)'].sum() + public_kwh

        calc = Calculator(_datas,
                          month=self.month,
                          contract="단일").set_households().set_apt(kwh=apt).set_bill()
        _pri_public_won = round(calc.public_won / calc.household_count)

        elec_rate_col_burden.append(pri_public_won / _pri_public_won)

    elec_rate_col_burden = np.array(elec_rate_col_burden)

    plt.figure(figsize=(16, 8))

    datas = [basic_col_burden, elec_rate_col_burden]
    titles = ['기본요금 수거부담율', '전력량요금 수거부담율']

    for idx, data in enumerate(datas):
        title = titles[idx]
        ax = plt.subplot(2, 1, idx+1)

        ax.bar(range(self.calc.household_count), data, color='g')
        ax.set_title(title)

    plt.show()

    return basic_col_burden, elec_rate_col_burden
