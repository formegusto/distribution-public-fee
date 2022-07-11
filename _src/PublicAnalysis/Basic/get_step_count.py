import pandas as pd
import numpy as np


def get_step_count(self):
    BASIC = self.analyzer.calc.households[0].BASIC
    step_list = np.array(
        [_.step_count for _ in self.analyzer.calc.households])
    steps, counts = np.unique(step_list, return_counts=True)

    result = pd.DataFrame()
    result['누진단계'] = steps
    result['가구 수'] = counts
    result['기본요금'] = BASIC[:steps.size]
    result['토탈'] = result['가구 수'] * result['기본요금']

    result.set_index("누진단계", inplace=True)
    result.loc['세대부 토탈'] = result.sum(axis=0)

    return result
