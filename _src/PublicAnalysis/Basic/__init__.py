from .get_percentage import get_percentage
from .get_step_count import get_step_count


class Basic:
    def __init__(self, analyzer):
        self.analyzer = analyzer


Basic.get_percentage = get_percentage
Basic.get_step_count = get_step_count
