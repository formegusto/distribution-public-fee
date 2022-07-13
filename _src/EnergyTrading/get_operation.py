from .utils import reactive_price_by_trade_qty


def get_buyer_profit(self, row, qty_kwh, price):
    usage_after_trade = row['usage (kWh)'] - qty_kwh
    price_margin = self.ELEC[usage_after_trade > self.step][-1] - price
    return price_margin * qty_kwh


def get_result(self, row, idx, trade_unit):
    trade_price_won = reactive_price_by_trade_qty(
        trade_unit,
        row['usage (kWh)'],
        self.step,
        self.ELEC)
    return [idx, trade_unit, trade_price_won, self.get_buyer_profit(row, trade_unit, trade_price_won)]
