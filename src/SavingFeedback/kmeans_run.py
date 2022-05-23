import numpy as np
import pandas as pd
from ..KMeans import KMeans


def kmeans_run(self):
    if self.datas is None:
        print("select_month(month:number) executing required.")
    kmeans = KMeans(datas=self.datas.T.values, ver=3)
    kmeans.fit()
    kmeans.sorting()

    self.kmeans = kmeans
    self.set_group()
