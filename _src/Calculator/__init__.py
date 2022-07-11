
from _src.Calculator.common.get_season_kr import get_season_kr

from .set_households import set_households
from .set_apt import set_apt
from .set_bill import set_bill


class Calculator:
    def __init__(self, datas, month, contract="단일"):
        if contract != "단일" and contract != "종합":
            raise ValueError("only 단일 or 종합")
        self.datas = datas
        self.household_count = len(datas)
        self.household_kwh = datas['usage (kWh)'].sum()
        self.contract = contract
        self.season = get_season_kr(month)


Calculator.set_households = set_households
Calculator.set_apt = set_apt
Calculator.set_bill = set_bill
