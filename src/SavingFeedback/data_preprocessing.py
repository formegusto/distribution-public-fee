import pandas as pd
import datetime as dt


def data_preprocessing(self, xlsx):
    indexes = xlsx.iloc[3:].apply(lambda x: dt.datetime(x[1],
                                                        x[2],
                                                        x[3],
                                                        x[4],
                                                        x[5]), axis=1).values
    columns = xlsx.iloc[:, 7:].apply(
        lambda x: "{}-{}-{}".format(x[0], x[1], x[2])).values
    datas = xlsx.iloc[3:, 7:].to_numpy()

    self.datas = pd.DataFrame(datas, columns=columns, index=indexes)
