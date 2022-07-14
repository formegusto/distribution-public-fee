import numpy as np
from .common import STEP_LIMITS_HOUSEHOLD, STEP, ELEC, get_season_kr


def set_init(self):
    self.season = get_season_kr(self.month)
    # usageBorder
    self.step = STEP[self.season]
    self.step_limit = STEP_LIMITS_HOUSEHOLD[self.season]
    self.ELEC = ELEC["단일"]

    _supply_filtered = self.datas[self.datas['usage (kWh)']
                                  < self.step_limit[0]]['usage (kWh)'].values
    self.sum_of_supply_kwh = np.sum(self.step_limit[0] - _supply_filtered)
