import numpy as np


def simulate(self):
    targets = self.datas.copy()
    _sum_of_supply_kwh = self.sum_of_supply_kwh
    result = list()

    cnt = 0
    while _sum_of_supply_kwh > 0:
        tmp_trade_unit = _sum_of_supply_kwh if _sum_of_supply_kwh < self.trade_unit else self.trade_unit

        buyer_idx = self.buyer_selection_by_list_idx(targets, tmp_trade_unit)
        result.append(
            self.get_result(targets.iloc[buyer_idx], buyer_idx, tmp_trade_unit)
        )

        targets.at[buyer_idx, 'usage (kWh)'] -= tmp_trade_unit
        _sum_of_supply_kwh -= tmp_trade_unit
        cnt += 1

    self.result = np.array(result)
