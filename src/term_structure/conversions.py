import numpy as np


def simple_rate_to_df(
        rate: float,
        t: float
) -> float:
    """ Converting money-market rate into discount factor """
    return 1.0 / (1.0 + rate * t)


def zero_to_df(
        zero_rate: float,
        t: float
) -> float:
    """ Converting continuously compounded zero-rate into discount factor """
    return np.exp(-zero_rate * t)


def df_to_zero(
        df: float,
        t: float
) -> float:
    """ Converting discount factor into continuously compounded zero-rate """
    return -np.log(df) / t
