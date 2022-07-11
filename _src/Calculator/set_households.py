from .models import Household


def set_households(self):
    households = list()
    for idx, row in self.datas.iterrows():
        household = Household(
            row['name'], row['usage (kWh)']).set_rate(self.contract, self.season)
        households.append(household)
    self.households = households

    return self
