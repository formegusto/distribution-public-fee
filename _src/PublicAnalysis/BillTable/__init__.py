import numpy as np
import pandas as pd


class BillTable:
    def __init__(self, analyzer, kwh=None, public_percentage=None):
        analyzer.set_calc(kwh=kwh, public_percentage=public_percentage)
        self.analyzer = analyzer

    @property
    def get(self):
        calc = self.analyzer.calc
        apt = calc.apt_obj
        households = calc.households

        apt_rates = np.array(list(apt.to_rate_dict(
            mul=calc.household_count).values()))
        household_rates = np.array(
            [list(_.to_rate_dict().values()) for _ in households]).sum(axis=0)
        public_rates = (apt_rates[:-1] - household_rates[:-1])
        public_rates = np.append(public_rates, public_rates.sum())

        keys = list(apt.to_rate_dict().keys())
        real_public_rates = np.zeros(len(keys))
        real_public_rates[-1] = apt_rates[-1] - household_rates[-1]

        public_err = np.zeros(len(keys))
        public_err[-1] = np.abs(public_rates[-1] - real_public_rates[-1])

        rates = np.append(apt_rates, household_rates, axis=0)
        rates = np.append(rates, public_rates, axis=0)
        rates = np.append(rates, real_public_rates, axis=0)
        rates = np.append(rates, public_err, axis=0).reshape(-1,
                                                             len(keys))

        # rates
        return pd.DataFrame(rates, columns=keys, index=['APT', '세대부', '공용부', '실제 공용부 값', '오차'], dtype=np.int)
