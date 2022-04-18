import math as mt
import numpy as np
import pandas as pd
from src.crs.models import ManagementOffice, Household


def get_step(kWh, fee):
    for step, _f in enumerate(fee):
        if kWh <= _f[1]:
            return step


def get_col_list(kWh, fee):
    col_list = list()
    _usage = kWh
    for step, _f in enumerate(fee):
        collect_rate = _f[1] - _f[0]

        # 2단계일 경우
        if step == 1:
            collect_rate += 1

        if kWh < _f[1]:
            if (step == 0) or (step == 1):
                _col = _usage % collect_rate
            else:
                _col = _usage
        else:
            _col = collect_rate

        if _usage == 0:
            col_list.append(0)
        else:
            col_list.append(_col)
            _usage -= _col

    return col_list


class PublicPredictor:
    def __init__(self,
                 APT,
                 month_df,
                 month):
        self.month_df = month_df
        self.APT = APT
        self.households_count = len(month_df)
        self.APT_mean = round(APT / self.households_count)
        self.households_mean = round(month_df['usage (kWh)'].mean())

        # 청구서 초기화
        self.calc = ManagementOffice(
            month=month,
            households=month_df,
            APT=APT,
            contract="단일계약"
        )
        self.fee = self.calc.fee

        # 최소 kWh
        min_kWh = 0
        while True:
            h_test = Household(
                name="min chk test",
                kwh=min_kWh,
                contract=self.fee,
                contract_name="단일계약"
            )
            elec_bill = h_test.elec_bill
            if elec_bill > 1000:
                min_kWh -= 1
                break
            else:
                min_kWh += 1
        self.min_kWh = min_kWh

        self._month_df = month_df.copy()
        self.min_households = month_df[month_df['usage (kWh)'] <= min_kWh].copy(
        )
        self.month_df = month_df[month_df['usage (kWh)'] > min_kWh].copy()

        self.households_kWh = self.month_df['usage (kWh)'].sum()

        self.min_pay = 1000 * len(self.min_households)

    def setting_basic(self):
        households_basic = 0
        for step, _f in enumerate(self.fee):
            now_kWh = _f[0]
            if step < 2:
                next_kWh = self.fee[step + 1][0]
                count = len(self.month_df[(self.month_df['usage (kWh)'] >= now_kWh) &
                                          (self.month_df['usage (kWh)'] < next_kWh)])
            else:
                count = len(
                    self.month_df[self.month_df['usage (kWh)'] >= now_kWh])

            households_basic += (_f[2] * count)

        apt_step = get_step(self.APT_mean, self.fee)
        apt_basic = self.fee[apt_step][2] * self.households_count

        self.basic = apt_basic - households_basic

    def setting_elec_rate(self):
        # 1. 공용부 전력량 요금 시작 요금 셋팅
        # 공용부가 0kWh 일 때 수거 해야 하는 사용량
        mean_households_col_list = np.array(
            get_col_list(self.households_mean, self.fee))
        mean_households_col_list *= self.households_count

        # 기본적으로 가구들이 수거해가는 전력 사용량
        households_col_list = np.array([])
        for idx, row in self.month_df.iterrows():
            name, usage = row

            households_col_list = np.append(households_col_list,
                                            get_col_list(usage, self.fee))
        households_col_list = households_col_list.reshape(-1, 3).sum(axis=0)

        # 오차 계산
        err_households_col_list = mean_households_col_list - households_col_list

        # 공용부 전력량 요금 시작 요금 산출
        start_elec_rate = round(
            (err_households_col_list * np.array(self.fee)[:, 3]).sum())
        self.start_elec_rate = start_elec_rate

        # 2. 공용부 시작 누진단계 산출
        apt_col_list = get_col_list(self.APT_mean, self.fee)
        div_households_col_list = np.divide(
            mean_households_col_list, self.households_count).astype("int")
        public_col_list = apt_col_list - div_households_col_list

        elec_rate = round((public_col_list * np.array(self.fee)
                          [:, 3] * self.households_count).sum())

        self.elec_rate = elec_rate + start_elec_rate

    # 기후환경요금, 연료비조정액, 필수사용량보장공제
    def setting_rest(self):
        self.env = round((self.APT - self.households_kWh) * 5.3)
        self.fuel = round((self.APT - self.households_kWh) * 3)

        self.guarantee = 0
        for h in self.calc.households:
            if h.kwh > self.min_kWh:
                self.guarantee += h.guarantee

    @property
    def predict(self):
        self.setting_basic()
        self.setting_elec_rate()
        self.setting_rest()

        elec_bill = self.basic + self.elec_rate + self.env - \
            self.fuel + self.guarantee - self.min_pay

        vat = round(elec_bill * 0.1)
        fund = mt.floor(elec_bill * 0.037 * 0.1) * 10

        elec_bill += vat + fund
        elec_bill = mt.floor(elec_bill * 0.1) * 10

        real_elec_bill = self.calc.public_bill
        err = real_elec_bill - elec_bill

        rtn_sr = pd.Series({
            "실제 요금": real_elec_bill,
            "예측 요금": elec_bill,
            "오차": err
        })

        return rtn_sr
