import pandas as pd


def parallel_shock(
        curve: pd.DataFrame,
        shock_bps: float = 1.0
) -> pd.DataFrame:
    """
    Parallel shock across all maturities in the curve

    Formula:
        r_shocked(t) = r(t) + shock_bps * 1e-4
    """
    shock = shock_bps * 1e-4
    return curve + shock


def key_rate_shock(
        curve: pd.DataFrame,
        tenor: float,
        shock_bps: float = 1.0
) -> pd.DataFrame:
    """
    Single tenor (key-rate) shock in the curve
    """
    shocked_curve = curve.copy()
    shock = shock_bps * 1e-4

    if tenor not in shocked_curve.columns:
        raise ValueError(f'Tenor: {tenor} not found in the curve columns!')
    
    shocked_curve[tenor] += shock
    return shocked_curve


def multi_tenor_shock(
        curve: pd.DataFrame,
        shock_dict: dict
) -> pd.DataFrame:
    """
    Multi-tenor shock in the curve given by shock dictionary {tenor: shock_bps}
    """
    shocked_curve = curve.copy()

    for tenor, shock_bps in shock_dict.items():

        if tenor not in shocked_curve.columns:
            raise ValueError(f'Tenor: {tenor} not found in the curve columns!')
        
        shock = shock_bps * 1e-4

        shocked_curve[tenor] += shock

    return shocked_curve