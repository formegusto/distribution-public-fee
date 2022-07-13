from .utils import reactive_price_by_trade_qty


def buyer_selection_by_list_idx(self, targets, trade_unit):
    bnf_seller = targets.apply(lambda x:
                               (reactive_price_by_trade_qty(
                                   trade_unit,
                                   x['usage (kWh)'],
                                   self.step,
                                   self.ELEC
                               ) - self.ELEC[0]) * trade_unit, axis=1)

    return bnf_seller.argmax()
