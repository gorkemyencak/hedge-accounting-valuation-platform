import numpy as np
import pandas as pd


# generic forward-curve builder using proxy forwards
def build_forward_curve_from_discount_curve(
        discount_curve: pd.DataFrame
) -> pd.DataFrame:
    """
    Converting bootstrapped discount factors into forward curve using discrete approximate forwards

    Discrete Proxy Formula:
        f(t_i) = (ln DF(t_i-1) - ln DF(t_i)) / (t_i - t_i-1)
    """
    forward_curves = []

    times = discount_curve.columns.astype(float)

    for date, row in discount_curve.iterrows():
        
        dfs = row.values.astype(float)

        forwards = {}
        for t in range(1, len(times)):
            
            t1, t2 = times[t-1], times[t]
            df_t1, df_t2 = dfs[t-1], dfs[t]

            forward_rate = (np.log(df_t1) - np.log(df_t2)) / (t2 - t1)
            forwards[t2] = 100 * forward_rate

        forward_curves.append(
            pd.Series(
                forwards,
                name = date
            )
        ) 

    return pd.DataFrame(forward_curves)
