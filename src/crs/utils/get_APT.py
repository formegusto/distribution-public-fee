def get_APT(month_df, PUBLIC_PERCENTAGE):
    households_kWh = month_df['usage (kWh)'].sum()
    APT = round((households_kWh * 100) / (100 - PUBLIC_PERCENTAGE))

    return APT
