import numpy as np


def reactive_price_by_trade_qty(qty_trade, usage, step, elec_rate):
    _step = step[1:]
    min_usage = step[1]
    # 전력량요금
    elec_rate = elec_rate[:-1]

    if usage < min_usage:
        return 0

    # [demand2, demand1]
    demands = usage - _step
    demands = np.where(demands > 0, demands, 0)

    err_demands = demands[1] - demands[0]
    output_price = ((elec_rate[1] - elec_rate[0]) / err_demands) * qty_trade -\
        (elec_rate[1] * demands[0] - elec_rate[0] * demands[1]) / (err_demands)

    return output_price
