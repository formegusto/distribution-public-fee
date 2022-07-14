import numpy as np
import sys

# 기본요금
BASIC = dict({
    "단일": np.array([730, 1260, 6060]),
    "종합": np.array([910, 1600, 7300])
})
# 전력량요금
ELEC = dict({
    "단일": np.array([78.2, 147.2, 215.5]),
    "종합": np.array([88.3, 182.9, 275.6])
})
# 기후환경요금 단위
ENV = 7.3
# 연료비조정액
FUEL = 5
# FUEL = -3
# 필수사용량보장공제
GUARANTEE = dict({
    "단일": 0,
    "종합": 0
})

# 분할 계산 설정 참고 VALUE
STEP_LIMITS_HOUSEHOLD = dict({
    "기타": np.array([200, 200]),
    "여름": np.array([300, 150])
})

STEP = dict({
    "기타": np.array([0, 200, 400]),
    "여름": np.array([0, 300, 450])
})
