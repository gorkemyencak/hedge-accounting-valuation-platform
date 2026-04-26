import numpy as np
import pandas as pd


def payment_schedule(
        maturity: float,
        freq: int = 2
):
    """ 
    Returns payment times in year fractions 
    
    t_i = i / freq for i = 1, .., n and n = freq * T
    """
    n_payments = int(freq * maturity)
    t_i = np.arange(1, n_payments + 1) * (1 / freq)
    return t_i

# helper function to get discount factor
def get_df(
        df_curve: pd.DataFrame,
        t: float
):
    """
    Returns DF(t) for all dates for a given t
    """
    if t not in df_curve.columns:
        raise ValueError(f'DF({t}) not found in the discount curve!')
    return df_curve[t]


def fixed_leg_annuity(
        df_curve: pd.DataFrame,
        maturity: float,
        freq: int = 2
):
    """
    Returns annuity time series indexed by date

    Formula:
        A(t) = sum (DF(t_i) * delta)

        where delta = 1 / freq
    """
    times = payment_schedule(
        maturity = maturity,
        freq = freq
    )

    delta = 1 / freq

    annuity = 0.0

    for t in times:
        annuity += get_df(df_curve = df_curve, t = t) * delta

    return annuity


def floating_leg_pv(
        df_curve: pd.DataFrame,
        maturity: float
):
    """
    Returns PV of a floating-rate swap prices at par as a time series indexed by date 
    
    Formula:
        PV_float = 1 - DF(T)
    """
    df_T = get_df(
        df_curve = df_curve,
        t = maturity
    )

    return 1 - df_T

# par swap rate for a single maturity
def par_swap_rate(
        df_curve: pd.DataFrame,
        maturity: float,
        freq: int = 2
):
    """
    Assuming a swap value = 0, it returns the fixed coupon that makes both fixed- and floating-legs equal
    
    Formula: 
        K = Floating leg PV / Fixed leg Annuity   
            -> K(T) = (1 - DF(T)) / A(T) 
    """
    floating_leg = floating_leg_pv(
        df_curve = df_curve,
        maturity = maturity
    ) 
    
    fixed_leg = fixed_leg_annuity(
        df_curve = df_curve,
        maturity = maturity,
        freq = freq
    )

    return floating_leg / fixed_leg

# generic par swap curve
def par_swap_curve(
        df_curve: pd.DataFrame,
        maturities: list[float],
        freq: int = 2
):
    """
    Returns par swap curve for a given list of maturities
    """
    swap_curve = pd.DataFrame(index = df_curve.index)

    for T in maturities:
        swap_curve[T] = par_swap_rate(
            df_curve = df_curve,
            maturity = T,
            freq = freq
        )

    return swap_curve
