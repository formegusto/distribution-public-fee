def set_cont(self):
    self.cont_ = (self.meter_month['usage (kWh)'] /
                  self.meter_month['usage (kWh)'].sum()).values
