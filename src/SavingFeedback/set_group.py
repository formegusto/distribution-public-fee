import numpy as np
import pandas as pd


def set_group(self):
    group = np.column_stack([
        self.datas.columns,
        self.datas.sum(axis=0).astype(np.float).round().astype("int")
    ])
    group = np.column_stack([group, self.kmeans.labels_.astype("int")])
    self.group = pd.DataFrame(group, columns=['name', 'usage (kWh)', 'label'])
