def _get_usage(self):
    if self.mode == "time":
        return [round(_.sum()) for _ in self.simulations]
    elif self.mode == "day":
        return [round(sum([_.sum() for _ in sims]))
                for sims in self.simulations]


def result(self):
    sim_usage = _get_usage(self)
    self.new_group = self.group.copy()
    self.new_group['usage (kWh)'] = sim_usage
