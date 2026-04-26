import numpy as np
import pandas as pd


def build_zero_curve(df_curve: pd.DataFrame):
    """ 
    Converting discount factor curve into continuously compounded zero-rate curve 
    
    Formula:
        z(t) = - ln DF(t) / t
    """
    return -np.log(df_curve) / df_curve.columns.values


def build_forward_curve(df_curve: pd.DataFrame):
    """ 
    Converting discount factor curve into forward rate curve 
    
    Formula:
        f(t_1, t_2) = (ln DF(t_1) - ln DF(t_2)) / (t_2 - t_1)

        where t_1 < t_2
    """
    times = df_curve.columns.values

    forward_curve = []
    
    for date, row in df_curve.iterrows():
        
        forward_rates = {}

        for i in range(1, len(times)):

            t_1, t_2 = times[i-1], times[i]
            df_1, df_2 = row[t_1], row[t_2]

            forward_rates[t_2] = (
                (np.log(df_1) - np.log(df_2))
                / (t_2 - t_1)
            ) 
            
        forward_curve.append(
            pd.Series(
                forward_rates,
                name = date
            )
        )

    return pd.DataFrame(forward_curve)