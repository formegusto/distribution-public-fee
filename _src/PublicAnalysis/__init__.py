from .Basic import Basic
from .ElecRate import ElecRate
from .BillTable import BillTable
from .set_init import set_init
from .set_calc import set_calc
from .select_month import select_month
from .get_col_burden import get_col_burden


class PublicAnalysis:
    def __init__(self, file_path):
        self.file_path = file_path

    def set(self):
        self.basic = Basic(analyzer=self)
        self.elec_rate = ElecRate(analyzer=self)

    def set_bill_table(self, public_percentage=30):
        self.bill_table = BillTable(
            analyzer=self, public_percentage=public_percentage)


PublicAnalysis.set_init = set_init
PublicAnalysis.set_calc = set_calc
PublicAnalysis.select_month = select_month
PublicAnalysis.get_col_burden = get_col_burden
