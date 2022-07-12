from ..Calculator import Calculator
from .min_chk import min_chk
from .predict_rate import basic, elec_rate, env, \
    fuel, guarantee, min_won, elec_bill, \
    vat, fund, elec_bill_vat_fund


class PublicPredictor:
    def __init__(self, meter_month, month, contract="단일", kwh=None, public_percentage=None):
        self.calc = Calculator(meter_month,
                               month=month,
                               contract=contract).set_households().set_apt(kwh=kwh, public_percentage=public_percentage).set_bill()
        self.meter_month = meter_month
        self.month = month

    @property
    def predict(self):
        return [self.elec_bill_vat_fund, self.calc.public_won, abs(self.elec_bill_vat_fund - self.calc.public_won)]


PublicPredictor.min_chk = min_chk
PublicPredictor.basic = basic
PublicPredictor.elec_rate = elec_rate
PublicPredictor.min_won = min_won
PublicPredictor.env = env
PublicPredictor.fuel = fuel
PublicPredictor.guarantee = guarantee
PublicPredictor.elec_bill = elec_bill
PublicPredictor.vat = vat
PublicPredictor.fund = fund
PublicPredictor.elec_bill_vat_fund = elec_bill_vat_fund
