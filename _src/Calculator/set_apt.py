from .models import Household


def set_apt(self, kwh=None, public_percentage=None):
    if kwh is not None:
        self.apt = kwh
    elif public_percentage is not None:
        self.apt = round((self.household_kwh * 100) /
                         (100 - public_percentage))
    else:
        raise ValueError("need kwh or public_percentage params")

    self.public_kwh = self.apt - self.household_kwh
    self.apt_obj = Household("APT", round(
        self.apt / self.household_count)).set_rate(self.contract, self.season)

    return self
