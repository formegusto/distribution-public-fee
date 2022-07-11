import pandas as pd
import numpy as np


def col_table(self):
    # 1. 단계별, 수거 필요 사용량
    calc = self.analyzer.calc
    all_col_needed = calc.apt_obj.steps * calc.household_count

    # Household Steps
    household_steps = np.array([_.steps for _ in calc.households])
    household_steps

    # 2. 수거 사용량
    col_households = household_steps.sum(axis=0)
    # 음수 : 잉여 사용량, 양수 : 초과 수거 사용량
    col_arr = household_steps - calc.apt_obj.steps

    # 잉여 사용량
    surplus_col = col_arr.copy()
    surplus_col[surplus_col > 0] = 0
    surplus_col = np.abs(surplus_col.sum(axis=0))

    # 초과 수거 사용량
    excess_col = col_arr.copy()
    excess_col[excess_col < 0] = 0
    excess_col = excess_col.sum(axis=0)
    excess_col

    return pd.DataFrame(np.array([
        all_col_needed,
        col_households,
        surplus_col,
        excess_col
    ]).astype("int"), columns=['누진 1단계', '누진 2단계', '누진 3단계'],
        index=['수거 필요 사용량', ' 수거 사용량', '잉여 사용량', '초과수거 사용량'])
