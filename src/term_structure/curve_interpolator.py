import numpy as np
import pandas as pd


def log_linear_curve_interpolator(
        df_curve,
        _target_times
):
    """ 
    Log linear discount curve interpolator using coupon term structures 
    
    Formula:
        ln DF(t) = w * ln DF(t_1) + (1 - w) * ln DF(t_2)

        where t_1 < t < t_2
        and w = (t_2 - t) / (t_2 - t_1) 
    """
    target_times = np.array(_target_times, dtype = float)

    interp_curve = []

    for date, row in df_curve.iterrows():

        times = np.array(row.index, dtype = float)
        dfs = np.array(row.values, dtype = float)

        # ensure sorted times and dfs
        sorted_idx = np.argsort(times)
        times = times[sorted_idx]
        dfs = dfs[sorted_idx]

        log_dfs = np.log(dfs)
        dfs_interp = np.zeros_like(target_times, dtype = float)

        for i, t in enumerate(target_times):

            # extrapolation before start
            if t < times[0]:
                dfs_interp[i] = dfs[0]
                continue

            # extrapolation after end
            if t > times[-1]:
                dfs_interp[i] = dfs[-1]
                continue

            # finding start and end coupon times for interpolation
            idx_end = np.searchsorted(times, t)
            idx_start = idx_end - 1

            t1, t2 = times[idx_start], times[idx_end]
            log_df1, log_df2 = log_dfs[idx_start], log_dfs[idx_end]

            # weight
            w = (t2 - t) / (t2 - t1)

            log_df_t = (
                w * log_df1
                + (1 - w) * log_df2
            )

            dfs_interp[i] = np.exp(log_df_t)
        
        interp_curve.append(
            pd.Series(
                dfs_interp,
                index = target_times,
                name = date
            )
        )

    return pd.DataFrame(interp_curve)
