from .col_monitor import col_monitor
from .get_kwh import get_kwh, get_rolling
from .col_table import col_table


class ElecRate:
    def __init__(self, analyzer):
        self.analyzer = analyzer


ElecRate.col_monitor = col_monitor
ElecRate.get_kwh = get_kwh
ElecRate.col_table = col_table
ElecRate.get_rolling = get_rolling
