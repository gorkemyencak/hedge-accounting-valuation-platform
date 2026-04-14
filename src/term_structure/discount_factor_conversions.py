import numpy as np

def df_from_simple_rate(
        rate: float,
        yearfrac: float
) -> float:
    """
    Converts money-market rate into discount factor
    """
    # converting rate given in percent into decimal points
    r = rate / 100

    df = 1.0 / (1.0 + r * yearfrac)

    return df

def zero_rate_from_df(
        df: float,
        yearfrac: float
) -> float:
    """ Converts discount factor to contiuously compounded zero rate """
    zero_rate = -np.log(df) / yearfrac

    return zero_rate
