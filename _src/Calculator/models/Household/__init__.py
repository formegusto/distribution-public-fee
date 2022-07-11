from .set_rate import set_rate
from .calc import basic, elec_rate, env, fuel, guarantee, elec_bill, fund, vat, elec_bill_vat_fund


class Household:
    def __init__(self, name, kwh):
        self.name = name
        self.kwh = kwh

    def to_rate_dict(self, mul=1):
        return {
            "기본요금": self.basic * mul,
            "전력량요금": self.elec_rate * mul,
            "기후환경요금": self.env * mul,
            "연료비조정액": self.fuel * mul,
            "부가가치세": self.vat * mul,
            "전력산업기반기금": self.fund * mul,
            "최종청구금액": self.elec_bill_vat_fund * mul
        }


Household.set_rate = set_rate
Household.basic = basic
Household.elec_rate = elec_rate
Household.env = env
Household.fuel = fuel
Household.guarantee = guarantee
Household.elec_bill = elec_bill
Household.fund = fund
Household.vat = vat
Household.elec_bill_vat_fund = elec_bill_vat_fund
