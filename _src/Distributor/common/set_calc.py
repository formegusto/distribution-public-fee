from ...Calculator import Calculator
from ...PublicPredictor import PublicPredictor


def set_calc(self, contract="단일", kwh=None, public_percentage=None):
    self.calc = Calculator(self.meter_month,
                           month=self.month,
                           contract=contract).set_households().set_apt(kwh=kwh, public_percentage=public_percentage).set_bill()
    self.predictor_ = PublicPredictor(
        meter_month=self.meter_month, month=self.month, kwh=kwh, public_percentage=public_percentage).min_chk()
