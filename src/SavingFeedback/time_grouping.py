import numpy as np
import datetime as dt


def time_grouping(self, target, time_size, need_mean=True):
    times = [(start_time, start_time + (time_size-1))
             for start_time in range(0, 24, time_size)]

    time_index = self.datas.index
    time_group = list()

    for start_time, end_time in times:
        _condition = ((time_index.time >= dt.time(start_time, 0))
                      & (time_index.time <= dt.time(end_time, 0)))
        time_group.append(
            target[_condition].reshape(-1, time_size)
        )

    time_group = np.array(time_group)

    if need_mean:
        return time_group, time_group.mean(axis=1)
    else:
        return time_group
