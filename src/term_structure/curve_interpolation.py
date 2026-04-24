import numpy as np
import pandas as pd

from src.term_structure.conventions import Tenors_To_Yearfrac_Interpolation


def tenor_to_yearfrac(
        tenor: str
) -> float:
    """ Converting tenor string to year fraction """
    if tenor == 'ON':
        return 1/360
    if tenor.endswith('M'):
        return int(tenor[:-1]) / 12
    if tenor.endswith('Y'):
        return int(tenor[:-1])
    
    raise ValueError(f'Unknown tenor: {tenor}')


def build_coupon_structure(
        max_year = 30,
        freq = 2 
):
    """ Building coupon grid along the maturity based on coupon payment frequency """
    step = 1 / freq
    return np.arange(step, max_year + step, step)


def zero_rate_to_df(zero_curve_interp):
    """ Helper function to convert zero rates into discount factors """
    dfs =[]

    for date, row in zero_curve_interp.iterrows():

        dfs_row = np.exp(-row.values / 100 * row.index.values)

        dfs.append(
            pd.Series(
                dfs_row,
                index = row.index,
                name = date 
            )
        )
    
    return pd.DataFrame(dfs)


def interpolate_zero_curve(
        zero_curve_df,
        target_times
):
    """ Linear interpolation of zero rates to target maturities """
    interpolated_rows = []
    
    # Converting dataframe columns (tenors) into year fractions
    base_times = np.array(
        [tenor_to_yearfrac(t) for t in zero_curve_df.columns],
        dtype = float
    )

    # ensure sorted base times
    sort_idx = np.argsort(base_times)
    base_times = base_times[sort_idx]

    for date, row in zero_curve_df.iterrows():
        base_rates = row.values.astype(float)[sort_idx]

        interpolated_rates = np.interp(
            target_times,
            base_times,
            base_rates
        )

        interpolated_rows.append(
            pd.Series(
                interpolated_rates,
                index = target_times,
                name = date
            )
        )
    
    return pd.DataFrame(interpolated_rows)
