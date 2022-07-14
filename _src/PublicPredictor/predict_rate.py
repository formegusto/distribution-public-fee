import numpy as np
from ..Calculator.common.rate import ENV, FUEL


@property
def basic(self):
    col_needed = self.calc.apt_obj.basic * self.calc.household_count
    col_household = sum([
        _.basic if _.kwh > self.min_kwh else 0
        for _ in self.calc.households])

    return col_needed - col_household


@property
def elec_rate(self):
    ELEC = self.calc.households[0].ELEC
    elec_frame = np.zeros(ELEC.size)

    col_needed = self.calc.apt_obj.steps * self.calc.household_count
    col_household = np.array(
        [_.steps if _.kwh > self.min_kwh else elec_frame for _ in self.calc.households])
    col_household = col_household.sum(axis=0)

    remain_col = col_needed - col_household

    remain_col *= ELEC
    return remain_col.sum()


@property
def min_won(self):
    return sum([0 if _.kwh > self.min_kwh else 1000 for _ in self.calc.households])


@property
def env(self):
    return (self.calc.apt_obj.env * self.calc.household_count) \
        - sum([ENV * _.kwh if _.kwh > self.min_kwh else 0 for _ in self.calc.households])


@property
def fuel(self):
    return (self.calc.apt_obj.fuel * self.calc.household_count) \
        - sum([FUEL * _.kwh if _.kwh >
              self.min_kwh else 0 for _ in self.calc.households])


@property
def guarantee(self):
    apt_guarantee = self.calc.apt_obj.guarantee * self.calc.household_count
    return apt_guarantee - sum([_.guarantee if _.kwh > self.min_kwh else 0 for _ in self.calc.households])


@property
def elec_bill(self):
    # print(self.basic)
    # print(self.elec_rate)
    # print(self.env)
    # print(self.fuel)
    # print(self.guarantee)
    # print(self.min_won)
    return self.basic + self.elec_rate + self.env + \
        self.fuel - self.guarantee - self.min_won


@property
def vat(self):
    return round(self.elec_bill * 0.1)


@property
def fund(self):
    return np.floor(self.elec_bill * 0.037 * 0.1) * 10


@property
def elec_bill_vat_fund(self):
    return self.elec_bill + self.vat + self.fund
