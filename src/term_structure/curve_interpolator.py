import numpy as np
import pandas as pd


def log_linear_curve_interpolator(
        df_curve,
        target_times
):
    """ Log linear discount curve interpolator using coupon term structures """
    target_times = np.array(target_times)

    interp_curve = []

    for date, row in df_curve.iterrows():

        print(date)
        print(row)
        break

    return pd.DataFrame()
