# meter_month [name, 월 총 사용량] : 구성의 데이터 setting

import pandas as pd
import numpy as np


def select_month(self, month=1):
    meter_month = self.meter_year.loc[[month]]
    self.month = month
    self.meter_month = pd.DataFrame(np.column_stack([meter_month.columns, meter_month.values.flatten()]),
                                    columns=['name', 'usage (kWh)'])
