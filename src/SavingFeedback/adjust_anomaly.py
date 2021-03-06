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

        # 순차적인 라벨 구성을 위한 조정
        uni_labels = np.unique(new_labels)

        _new_labels = np.zeros(new_labels.size) - 1
        for idx, label in enumerate(uni_labels):
            _new_labels[new_labels == label] = idx

        _new_labels = _new_labels.astype("int")

        self.group['label'] = _new_labels

        self.kmeans.K = uni_labels.size
        self.kmeans.clusters_ = self.kmeans.clusters_[:uni_labels.size]
        self.kmeans.labels_ = _new_labels
        self.kmeans.next_setting()
