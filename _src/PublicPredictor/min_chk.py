from ..Calculator.models import Household

# 최저요금 지불 사용량 체크용


def min_chk(self):
    min_kwh = 0
    # while True:
    #     h_test = Household(
    #         name="min chk test",
    #         kwh=min_kwh,
    #     ).set_rate(contract=self.calc.contract,
    #                season=self.calc.season)
    #     elec_bill = h_test.elec_bill
    #     if elec_bill > 1000:
    #         min_kwh -= 1
    #         break
    #     else:
    #         min_kwh += 1
    self.min_kwh = min_kwh

    return self
