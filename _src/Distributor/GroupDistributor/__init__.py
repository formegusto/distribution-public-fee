from ..common import set_init, select_month, set_calc
from .set_cont import set_cont
from .distribute import distribute, distribute_table
from ...PublicAnalysis.BillTable import BillTable


class GroupDistributor:
    def __init__(self, file_path):
        self.file_path = file_path

    def set_bill_table(self, public_percentage=30):
        self.bill_table = BillTable(
            analyzer=self, public_percentage=public_percentage)


GroupDistributor.set_init = set_init
GroupDistributor.select_month = select_month
GroupDistributor.set_calc = set_calc
GroupDistributor.set_cont = set_cont
GroupDistributor.distribute = distribute
GroupDistributor.distribute_table = distribute_table
