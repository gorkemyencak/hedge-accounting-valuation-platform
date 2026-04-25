import pandas as pd


def merge_curves(
        short_curve,
        long_curve
):
    """ Merging short-end money market and long-end treasury curves into a single curve """

    merged = []

    for date in short_curve.index:

        short_row = short_curve.loc[date]
        long_row = long_curve.loc[date]

        full = pd.concat([short_row, long_row])
        full = full[~full.index.duplicated(keep='last')]

        merged.append(full)
    
    return pd.DataFrame(merged)
