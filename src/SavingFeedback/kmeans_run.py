from ..KMeans import KMeans
from ..TimeDivisionKMeans import TimeDivisionKMeans


def kmeans_run(self):
    if self.datas is None:
        print("select_month(month:number) executing required.")
    if self._type == "kmeans":
        kmeans = KMeans(datas=self.datas.T.values, ver=3)
        kmeans.fit()
    elif self._type == "tdkmeans":
        kmeans = TimeDivisionKMeans(datas=self.datas)
        kmeans.auto_fit(_type=self._tdtype)

    if self._type == "kmeans":
        kmeans.sorting()

    self.kmeans = kmeans
    self.set_group()
