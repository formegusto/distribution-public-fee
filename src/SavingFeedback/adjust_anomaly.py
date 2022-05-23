import numpy as np
from sklearn.tree import DecisionTreeClassifier as DTC


def adjust_anomaly(self):
    while True:
        anomaly = self.check_anomaly()

        if len(anomaly) == 0:
            break

        chk = np.isin(self.group['name'], anomaly['name'])
        unanomaly = self.group[
            ~chk
        ]

        target_cols = ['usage (kWh)']
        label_cols = ['label']

        X = unanomaly[target_cols].values.astype("int")
        y = unanomaly[label_cols].values.astype("int")

        dt = DTC()
        dt.fit(X, y)

        anomaly_X = anomaly[target_cols].values.astype("int")
        predict_label = dt.predict(anomaly_X)

        new_labels = self.group['label'].values.copy()
        new_labels[chk] = predict_label

        self.group['label'] = new_labels
