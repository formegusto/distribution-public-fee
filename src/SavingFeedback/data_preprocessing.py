import pandas as pd
import datetime as dt


def data_preprocessing(self, xlsx):
    # 1. xlsx parsing
    indexes = xlsx.iloc[3:].apply(lambda x: dt.datetime(x[1],
                                                        x[2],
                                                        x[3],
                                                        x[4],
                                                        x[5]), axis=1).values
    columns = xlsx.iloc[:, 7:].apply(
        lambda x: "{}-{}-{}".format(x[0], x[1], x[2])).values
    datas = xlsx.iloc[3:, 7:].to_numpy()
    df = pd.DataFrame(datas, columns=columns, index=indexes)

    # 2. convert 1hours
    m_15 = df.to_numpy().T
    m_15 = m_15.reshape(-1, round(len(df) / 4), 4)
    m_60 = m_15.sum(axis=2)
    self.m_60 = pd.DataFrame(
        m_60.T, columns=df.columns, index=df.index[::4])
