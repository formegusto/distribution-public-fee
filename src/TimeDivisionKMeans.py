from .KMeans import KMeans


class TimeDivisionKMeans:
    def __init__(self, datas, division_size=3):
        if (24 % division_size) != 0:
            raise ValueError("only 24 % division_size == 0 !!")

        self.datas = datas.reshape(-1, round(len(datas.T) /
                                   division_size), division_size)
        r, c = datas.shape
        self.division_round = round(c / division_size)

    def fit(self):
        self.kmeans_ = list()

        for _round in range(self.division_round):
            kmeans = KMeans(datas=self.datas[:, _round])
            kmeans.fit(logging=False)
            kmeans.sorting()

            self.kmeans_.append(kmeans)
            if (_round % 10 == 0) or \
                    (_round + 1 == self.division_round):
                print("{}/{} - ECV:{}%".format(_round+1,
                      self.division_round, round(kmeans.ecv * 100)))
