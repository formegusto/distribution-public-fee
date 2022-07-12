import numpy as np
import pandas as pd


def distribute(self, only_value=True, private=False):
    cont, counts = np.unique(self.cont_, return_counts=True)

    basic = round(self.predictor_.basic / self.cont_.size)
    elec_rate = round(self.predictor_.elec_rate / self.cont_.size)

    pri_basics = (basic * counts * cont / counts).round()
    pri_elec_rates = (elec_rate * counts * cont / counts).round()

    basics = np.zeros(self.cont_.size) if basic <= 0 else np.array(
        [pri_basics[_] for _ in self.groups_])
    elec_rates = np.zeros(self.cont_.size) if elec_rate <= 0 else np.array(
        [pri_elec_rates[_] for _ in self.groups_])

    rest = self.calc.public_won - basics.sum() - elec_rates.sum()
    rest_private = (rest / self.cont_.size)
    elec_bills = basics + elec_rates
    elec_bills += rest_private

    if only_value:
        return elec_bills.astype(np.float).round()
    else:
        return {
            "기본요금": basics,
            "전력량요금": elec_rates,
            "나머지": rest_private,
            "최종 공용부 요금": elec_bills.astype(np.float).round()
        }


def distribute_table(self, private=False):
    columns = ['가구명', '공용부요금', '기여도', '기여도 기본요금',
               '기여도 전력량요금', '나머지 요금', '기여도 요금', '오차']

    # 가구명
    datas = self.meter_month['name'].values

    # 기존 공용부 요금
    public_private = round(self.calc.public_won / self.cont_.size)
    datas = np.column_stack([datas, np.full(self.cont_.size, public_private)])

    # 기여도
    datas = np.column_stack([datas, self.cont_])

    # 기여도 공용부 요금
    distribute_result = distribute(self, only_value=False, private=private)
    # 기본요금
    datas = np.column_stack([datas, distribute_result['기본요금']])
    # 전력량요금
    datas = np.column_stack([datas, distribute_result['전력량요금']])
    # 나머지 요금
    datas = np.column_stack(
        [datas, np.full(self.cont_.size, distribute_result['나머지'])])
    # 기여도 요금
    datas = np.column_stack([datas, distribute_result['최종 공용부 요금']])

    # 오차
    datas = np.column_stack(
        [datas, public_private - distribute_result['최종 공용부 요금']])

    return pd.DataFrame(datas, columns=columns)
