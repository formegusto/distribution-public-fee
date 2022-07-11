from ..common import set_init


class SimpleDistributor:
    def __init__(self, file_path):
        self.file_path = file_path


SimpleDistributor.set_init = set_init
