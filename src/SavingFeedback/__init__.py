from .data_preprocessing import data_preprocessing


class SavingFeedback:
    def __init__(self, xlsx):
        self.data_preprocessing(xlsx)


SavingFeedback.data_preprocessing = data_preprocessing
