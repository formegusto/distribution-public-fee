import numpy as np
import pandas as pd
import datetime as dt

# instance property initializer
# meter_15 : 15분 meter 데이터
# meter_year : 12년치 1달 총 사용량


def set_init(self):
    xlsx = pd.read_excel(self.file_path, header=None,
                         skiprows=2, engine='openpyxl')

    months = np.arange(1, 13)

    # 가구명 리스트
    names = xlsx.iloc[:3, 7:]
    names = names.apply(lambda x: "-".join(x.values.astype("str"))).to_numpy()

    # 날짜 리스트
    dates = xlsx.iloc[3:, 1:6]
    dates = dates.apply(lambda x: dt.datetime(
        x[1], x[2], x[3], x[4], x[5]), axis=1).to_numpy()

    usages = xlsx.iloc[3:, 7:].to_numpy()
    usages = np.where(usages == "-", 0, usages)

    # 15분 meter pattern
    self.meter_15 = pd.DataFrame(
        usages, columns=names, index=dates, dtype=np.float)

    meter_year = np.array([])

    for month in months:
        month_meter = self.meter_15[self.meter_15.index.month == month].to_numpy(
        )
        meter_year = np.append(meter_year, month_meter.sum(axis=0).round())

    # 한달 meter sum
    self.meter_year = pd.DataFrame(meter_year.reshape(-1, names.size),
                                   columns=names,
                                   index=months, dtype=np.int)
