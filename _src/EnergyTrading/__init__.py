import pandas as pd
from .simulate import simulate
from .set_init import set_init
from .buyer_selection_by_list_idx import buyer_selection_by_list_idx
from .get_operation import get_buyer_profit, get_result
from .set_settlement_infos import set_settlement_infos


class EnergyTrading:
    def __init__(self, file_path, month, trade_unit=5):
        self.file_path = file_path
        self.month = month
        self.trade_unit = trade_unit
        self.datas = pd.read_csv(file_path)
        self.datas.columns = ['name', 'usage (kWh)']


EnergyTrading.set_init = set_init
EnergyTrading.simulate = simulate
EnergyTrading.buyer_selection_by_list_idx = buyer_selection_by_list_idx
EnergyTrading.get_buyer_profit = get_buyer_profit
EnergyTrading.get_result = get_result
EnergyTrading.set_settlement_infos = set_settlement_infos
