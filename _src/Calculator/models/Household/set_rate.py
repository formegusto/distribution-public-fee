import numpy as np
from ...common import BASIC, ELEC, STEP_LIMITS_HOUSEHOLD, GUARANTEE


def set_rate(self, contract, season):
    self.BASIC = BASIC[contract]
    self.ELEC = ELEC[contract]
    self.GUARANTEE = GUARANTEE[contract]

    STEP_LIMITS = STEP_LIMITS_HOUSEHOLD[season]
    _kwh = self.kwh
    steps = np.zeros(self.ELEC.size)
    step = 0

    while _kwh > 0:
        if step == 2:
            steps[step] = _kwh
            _kwh -= _kwh
        else:
            err = _kwh - STEP_LIMITS[step]
            if err < 0:
                steps[step] = _kwh
                _kwh -= _kwh
            else:
                steps[step] = STEP_LIMITS[step]
                _kwh -= STEP_LIMITS[step]
        step += 1
    self.steps = steps
    self.step_count = np.where(steps != 0)[0]
    if self.step_count.size == 0:
        self.step_count = 0
    else:
        self.step_count = self.step_count[-1]

    return self
