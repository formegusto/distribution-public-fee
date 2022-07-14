import numpy as np
from ...common.rate import ENV, FUEL


@property
def basic(self):
    _basics = self.BASIC[self.steps != 0]
    if _basics.size == 0:
        return self.BASIC[0]
    else:
        return _basics.max()


@property
def elec_rate(self):
    return np.floor((self.ELEC * self.steps).sum())


@property
def env(self):
    return np.floor(self.kwh * ENV)


@property
def fuel(self):
    return np.floor(self.kwh * FUEL)


# min_bill = 1000
min_bill = 0


@property
def elec_bill(self):
    _elec_bill = self.basic + self.elec_rate + self.env + self.fuel - self.guarantee
    if _elec_bill < min_bill:
        return 1000
    else:
        return _elec_bill


@property
def guarantee(self):
    if self.kwh <= 200:
        return self.GUARANTEE
    else:
        return 0


@property
def vat(self):
    return round(self.elec_bill * 0.1)


@property
def fund(self):
    return np.floor(self.elec_bill * 0.037 * 0.1) * 10

# 개인 사용량 요금 + 부가가치세 + 전력산업기반기금


@property
def elec_bill_vat_fund(self):
    return np.floor((self.elec_bill + self.vat + self.fund) * 0.1) * 10
