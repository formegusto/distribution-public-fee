import numpy as np


def spread(self, target):
    if self.mode == "time":
        return target.flatten()
    elif self.mode == "day":
        _target = np.array([], dtype=np.datetime64)
        for _tar in target:
            _target = np.append(_target, _tar)
        return _target
