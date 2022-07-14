import pandas as pd
import numpy as np
from .model import Household


def set_settlement_infos(self):
    # Original 가격 구하기
    household_count = len(self.datas)

    total_household_usage = self.datas['usage (kWh)'].sum()
    public_usage = total_household_usage * 0.25

    APT = total_household_usage + public_usage
    APT_mean = APT / household_count
    apt = Household(name='apt', kwh=APT_mean).set_rate("단일", self.season)

    original_households_charge = self.datas.apply(
        lambda x: Household(name=x['name'], kwh=x['usage (kWh)'])
        .set_rate("단일", self.season).elec_rate, axis=1)\
        .values
    self.datas['전력량요금'] = original_households_charge
    whole_apt_charge = apt.elec_rate * household_count
    households_charge = original_households_charge.sum()
    public_charge = whole_apt_charge - households_charge
    self.datas['공용부요금'] = public_charge / household_count

    # 거래 정보
    result_df = pd.DataFrame(self.result, columns=[
                             "index", "거래수량", "거래액수(1kWh)", "이익"])
    total_trade_energy = result_df['거래수량'].sum()
    total_sales = (result_df['거래액수(1kWh)'] * result_df['거래수량']).sum()
    # 판매자들이 1kWh당 받을 수 있는 가격
    pri_total_sales = total_sales / total_trade_energy
    trade_result = pd.DataFrame(np.append(self.datas[['name', 'usage (kWh)']].values, np.zeros((self.datas.index.size, 4)), axis=1),
                                columns=['name', 'usage (kWh)', '구매수량', '구매가격', '판매수량', '판매가격'])

    result_df['총 거래액수'] = result_df['거래액수(1kWh)'] * result_df['거래수량']
    group_result = result_df.groupby('index').sum()[['거래수량', '총 거래액수']]

    # 구매자 정보
    trade_result.loc[group_result.index.values, '구매수량'] = group_result['거래수량']
    trade_result.loc[group_result.index.values,
                     '구매가격'] = group_result['총 거래액수']

    # 판매자 정보
    seller_datas = trade_result[trade_result['usage (kWh)'] < self.step_limit[0]].copy(
    )
    trade_result.loc[seller_datas.index,
                     '판매수량'] = self.step_limit[0] - seller_datas['usage (kWh)']
    trade_result.loc[seller_datas.index,
                     '판매가격'] = trade_result.loc[seller_datas.index, '판매수량'] * pri_total_sales

    self.trade_result = trade_result

    # 사용량 : 판매한 애들은 올라가고, 구매한 애들은 내려감
    simulation_datas = self.datas[['name',
                                  'usage (kWh)']].copy()

    simulation_datas['usage (kWh)'] += trade_result['판매수량']
    simulation_datas['usage (kWh)'] -= trade_result['구매수량']

    mod_households_charge = simulation_datas.apply(
        lambda x: Household(name=x['name'], kwh=x['usage (kWh)'])
        .set_rate("단일", self.season).elec_rate, axis=1)\
        .values
    mod_public_charge = whole_apt_charge - mod_households_charge.sum()
    simulation_datas['전력량 요금'] = self.datas['전력량요금']
    simulation_datas['거래 전력량요금(거래가격 적용 X)'] = mod_households_charge
    simulation_datas['공용부요금(거래 적용 X)'] = mod_public_charge / household_count
    simulation_datas['거래 전력량요금(거래 적용 O)'] = mod_households_charge

    # 거래가격 적용 : 판매한 애들은 내려가고, 구매한 애들은 올라감
    simulation_datas['거래 전력량요금(거래 적용 O)'] -= trade_result['판매가격'].values
    simulation_datas['거래 전력량요금(거래 적용 O)'] += trade_result['구매가격'].values

    simulation_datas['공용부요금(거래 적용 O)'] = (
        whole_apt_charge - simulation_datas['거래 전력량요금(거래 적용 O)'].sum()) / household_count

    simulation_datas['차이'] = simulation_datas['전력량 요금'] - \
        simulation_datas['거래 전력량요금(거래 적용 O)']

    simulation_datas.loc[self.datas.sort_values(
        by=['usage (kWh)'], ascending=False).index]
    self.simulation_datas = simulation_datas
