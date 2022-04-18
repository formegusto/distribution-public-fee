from src.crs.models import ManagementOffice
import pandas as pd
import math as mt
import numpy as np


def get_analysis_df(_month, month_df, APT):
    calc = ManagementOffice(
        month=_month,
        households=month_df,
        APT=APT,
        contract="단일계약"
    )
    apt = calc.apart

    apt_pd = pd.DataFrame(
        columns=['기본요금', '전력량요금', '기후환경요금', '연료비 조정액', 'VAT', '전력산업기반기금', '기본요금 + 전력량요금'])

    apt_pd = apt_pd.append(
        pd.Series({
            "기본요금": apt.basic,
            "전력량요금": apt.elec_rate,
            "기후환경요금": apt.env,
            "연료비 조정액": apt.fuel * -1,
            'VAT': round(calc.elec_bill * 0.1 / len(month_df)),
            '전력산업기반기금': "x",
            '기본요금 + 전력량요금': "x"
        }, name="가구 수 곱 적용 전"))
    apt_pd = apt_pd.append(
        pd.Series({
            "기본요금": apt.basic * len(month_df),
            "전력량요금": apt.elec_rate * len(month_df),
            "기후환경요금": apt.env * len(month_df),
            "연료비 조정액": apt.fuel * len(month_df) * -1,
            'VAT': round(calc.elec_bill * 0.1),
            '전력산업기반기금': round(mt.floor(calc.elec_bill * 0.037 * 0.1) * 10),
            '기본요금 + 전력량요금': apt.basic * len(month_df) + apt.elec_rate * len(month_df)
        }, name="가구 수 곱 적용 후"))

    households = calc.households

    households_pd = pd.DataFrame(
        columns=['기본요금', '전력량요금', '기후환경요금', '연료비 조정액', 'VAT', '전력산업기반기금', '기본요금 + 전력량요금'])

    for household in households:
        households_pd = households_pd.append(
            pd.Series({
                "기본요금": household.basic,
                "전력량요금": household.elec_rate,
                "기후환경요금": household.env,
                "연료비 조정액": household.fuel * -1,
                'VAT': int(household.vat),
                '전력산업기반기금': int(household.fund),
                '기본요금 + 전력량요금': household.basic + household.elec_rate
            }, name=household.name))

    analysis_pd = pd.concat([apt_pd.loc['가구 수 곱 적용 후'], households_pd.sum(
    ), apt_pd.loc['가구 수 곱 적용 후'] - households_pd.sum()], axis=1).T
    analysis_pd.index = ['APT', '세대부', '공용부']

    analysis_pd.loc['세대부 / APT percentage'] = (
        (analysis_pd.loc['세대부'] / analysis_pd.loc['APT']) * 100).astype(np.float64).round().astype("int")
    analysis_pd.loc['공용부 / APT percentage'] = (
        (analysis_pd.loc['공용부'] / analysis_pd.loc['APT']) * 100).astype(np.float64).round().astype("int")

    return households_pd, analysis_pd
