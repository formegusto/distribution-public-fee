from src.crs.models import ManagementOffice
import pandas as pd


def print_apt_info(_month, month_df, APT):
    households_kWh = month_df['usage (kWh)'].sum()
    public_kWh = APT - households_kWh
    PUBLIC_PERCENTAGE = round(public_kWh / APT * 100)

    calc = ManagementOffice(
        month=_month,
        households=month_df,
        APT=APT,
        contract="단일계약"
    )
    apt = calc.apart

    apt_bill = calc.bill
    households_bill = calc.bill - calc.public_bill
    public_bill = calc.public_bill

    print("APT : {}kWh (평균 사용량: {}kWh)".format(format(APT, ","), apt.kwh))
    print("\t\t{}원".format(format(apt_bill, ",")))
    print("세대부 : {}kWh ({}%) (평균 사용량: {}kWh)".format(format(households_kWh, ","),
                                                     100 - PUBLIC_PERCENTAGE,
                                                     round(month_df['usage (kWh)'].mean(
                                                     ))
                                                     ))
    print("\t\t{}원".format(format(households_bill, ",")))
    print("공용부 : {}kWh ({}%) (평균 사용량: {}kWh)".format(format(public_kWh, ","), PUBLIC_PERCENTAGE,
                                                     round(public_kWh / len(month_df))))
    print("\t\t{}원".format(format(public_bill, ",")))
