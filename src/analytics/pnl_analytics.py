import pandas as pd

from src.portfolio.swap_portfolio import SwapPortfolio

"""
This module will bridge the Risk into PnL of a swap portfolio

Actual PnL = Explained PnL + Residual

where:
    - Actual PnL -> full repricing
    - Explained PnL -> rate move x DV01
    - Residual -> model/interpolation error
"""

def compute_daily_pnl(
        portfolio_npv: pd.DataFrame
) -> pd.DataFrame:
    """ 
    Computing daily PnL fom portfolio NPV time series

    Formula:
        Actual_PnL_t = V_t - V_t-1

        where:
            V_t: portfolio value at day t
    """
    pnl = portfolio_npv.diff()
    pnl.columns = ['Actual_PnL']

    return pnl.dropna().round(1)


def compute_parallel_rate_change(
        zero_curve: pd.DataFrame
) -> pd.DataFrame:
    """ 
    Computing daily parallel rate shift of the curve

    Formula:
        Δr_t = mean(Δr_tenors) 
        
        where:
        Δr_tenors: Δr_t - Δr_t-1    -> avg. rate change across all tenors (parallel-shift assumption)
    """
    # daily changes for each tenor
    rate_changes = zero_curve.diff()

    # average move across tenors
    parallel_changes = rate_changes.mean(axis = 1)

    return pd.DataFrame(
        parallel_changes,
        columns = ['Rate_change']
    ).dropna()


def dv01_explained_pnl(
        portfolio_dv01: pd.DataFrame,
        rate_changes: pd.DataFrame
) -> pd.DataFrame:
    """
    Compute DV01-based explained PnL

    Formula:
        Explained_PnL_t = DV01_t-1 x Δr_t_bps 

        where:
            Δr_t_bps = Δr_t * 10000     -> since DV01 is per 1 bp move
    """
    # align indices
    dv01_lagged = (
        portfolio_dv01
        .shift(periods = 1)
        .dropna()
    )

    rate_changes = rate_changes.loc[dv01_lagged.index]

    # converting rate changes into bps
    rate_changes_bps = rate_changes * 10000

    # compute explained PnL
    explained_pnl = pd.DataFrame(
        {
            'Explained_PnL': dv01_lagged.iloc[:, 0] * rate_changes_bps.iloc[:, 0]
        }
    )
    
    return explained_pnl.round(1)


def compute_residual_pnl(
        actual_pnl: pd.DataFrame,
        explained_pnl: pd.DataFrame
) -> pd.DataFrame:
    """
    Formula:
        Residual = Actual_PnL - Explained_PnL
    """
    # joining actual and explained pnl into a signle dataframe
    pnl = actual_pnl.join(
        explained_pnl,
        how = 'inner'
    )
    
    # computing residual PnL
    pnl['Residual_PnL'] = pnl['Actual_PnL'] - pnl['Explained_PnL']

    return pnl