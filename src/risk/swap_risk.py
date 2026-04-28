import numpy as np
import pandas as pd

from src.term_structure.conversions import (
    zero_to_discount_curve,
    discount_to_zero_curve
)

from src.risk.rate_shocks import (
    parallel_shock,
    key_rate_shock,
    multi_tenor_shock
)

from src.pricing.swap_pricing_engine import swap_npv

"""
Approach:
        1) Price swap using base curve
        2) Parallel shift zero curve by +1 bp
        3) Reconstruct the discount curve
        4) Reprice swap
        5) Compute DV01

Formula:
    DV01 = (NPV_shocked - NPV_base) / shock
"""

# helper functions
def _dv01_parallel(
        df_curve: pd.DataFrame,
        maturity: float,
        fixed_rate: float,
        freq: int = 2,
        notional: float = 1_000_000,
        shock_bps: float = 1.0
):
    """ Computing DV01 of a payer swap (pay-fixed, receive float) using parallel shock """
    # Base NPV
    npv_base = swap_npv(
        df_curve = df_curve,
        maturity = maturity,
        fixed_rate = fixed_rate,
        freq = freq,
        notional = notional
    )

    # DF to zero curve
    zero_curve = discount_to_zero_curve(df_curve = df_curve)

    # Applying parallel shock to zero curve
    zero_curve_shocked = parallel_shock(
        curve = zero_curve,
        shock_bps = shock_bps
    )

    # Zero to discount curve
    discount_curve_shocked = zero_to_discount_curve(zero_curve = zero_curve_shocked)

    # Reprice swap with shocked curve
    npv_shocked = swap_npv(
        df_curve = discount_curve_shocked,
        maturity = maturity,
        fixed_rate = fixed_rate,
        freq = freq,
        notional = notional
    )

    # shock
    shock = shock_bps * 1e-4

    # Compute DV01
    dv01 = (npv_shocked - npv_base) / shock

    return dv01

def _dv01_key_rate(
        df_curve: pd.DataFrame,
        maturity: float,
        fixed_rate: float,
        key_rate_tenors: list[float],
        freq: int = 2,
        notional: float = 1_000_000,
        shock_bps: float = 1.0
):
    """ 
    Computing DV01 of a payer swap for each tenor independently using key-rate shocks 
    
    Returns a DV01 vector indexed by tenors
    """
    # Base NPV
    npv_base = swap_npv(
        df_curve = df_curve,
        maturity = maturity,
        fixed_rate = fixed_rate,
        freq = freq,
        notional = notional
    )

    # DF to zero curve
    zero_curve = discount_to_zero_curve(df_curve = df_curve)

    # shock
    shock = shock_bps * 1e-4
    dv01_dict = {}

    for tenor in key_rate_tenors:
        
        # Applying key-rate shock to each tenor in zero curve independently
        zero_curve_shocked = key_rate_shock(
            curve = zero_curve,
            tenor = tenor,
            shock_bps = shock_bps
        )

        # Zero to discount curve
        discount_curve_shocked = zero_to_discount_curve(zero_curve = zero_curve_shocked)

        # Repricing swap with shocked curve
        npv_shocked = swap_npv(
            df_curve = discount_curve_shocked,
            maturity = maturity,
            fixed_rate = fixed_rate,
            freq = freq,
            notional = notional
        )

        # Compute DV01 and store in DV01 dictionary
        dv01_dict[tenor] = (npv_shocked - npv_base) / shock
    
    return pd.Series(dv01_dict)

def _dv01_multi_tenor(
        df_curve: pd.DataFrame,
        maturity: float,
        fixed_rate: float,
        shock_dict: dict[float, float], # -> {tenor: shock_bps}  
        freq: int = 2,
        notional: float = 1_000_000
):
    """ Computing DV01 of a payer swap for multi-tenor scenario using multi-tenor shocks  """
    # Base NPV
    npv_base = swap_npv(
        df_curve = df_curve,
        maturity = maturity,
        fixed_rate = fixed_rate,
        freq = freq,
        notional = notional
    )

    # DF to zero curve
    zero_curve = discount_to_zero_curve(df_curve = df_curve)

    # Applying multi-tenor shock using shock dictionary
    zero_curve_shocked = multi_tenor_shock(
        curve = zero_curve,
        shock_dict = shock_dict
    )

    # Zero to discount curve
    discount_curve_shocked = zero_to_discount_curve(zero_curve = zero_curve_shocked)

    # Repricing swap with shocked curve
    npv_shocked = swap_npv(
        df_curve = discount_curve_shocked,
        maturity = maturity,
        fixed_rate = fixed_rate,
        freq = freq,
        notional = notional
    )

    # average shock bps for normalization
    avg_shock_bps = np.mean(list(shock_dict.values()))
    avg_shock = avg_shock_bps * 1e-4

    # Compute DV01
    dv01 = (npv_shocked - npv_base) / avg_shock

    return dv01


# public method
def swap_dv01_(
        df_curve: pd.DataFrame,
        maturity: float,
        fixed_rate: float,
        freq: int = 2,
        notional: float = 1_000_000,
        shock_bps: float = 1.0,
        shock_type: str = 'parallel',
        key_rate_tenors: list[float] | None = None,
        multi_tenor_dict: dict[float, float] | None = None
):
    """ 
    Main DV01 pipeline 
    
    Parameters:
        shock_type: 'parallel' | 'key_rate' | 'multi_tenor'
    """
    if shock_type == 'parallel':
        return _dv01_parallel(
            df_curve = df_curve,
            maturity = maturity,
            fixed_rate = fixed_rate,
            freq = freq,
            notional = notional,
            shock_bps = shock_bps
        )
    
    elif shock_type == 'key_rate':
        
        if key_rate_tenors is None:
            raise ValueError(f'key_rate_tenors must be provided!')

        return _dv01_key_rate(
            df_curve = df_curve,
            maturity = maturity,
            fixed_rate = fixed_rate,
            key_rate_tenors = key_rate_tenors,
            freq = freq,
            notional = notional,
            shock_bps = shock_bps
        )
    
    elif shock_type == 'multi_tenor':
        
        if multi_tenor_dict is None:
            raise ValueError(f'multi_tenor_dict must be provided!')

        return _dv01_multi_tenor(
            df_curve = df_curve,
            maturity = maturity,
            fixed_rate = fixed_rate,
            shock_dict = multi_tenor_dict,
            freq = freq,
            notional = notional
        )
    
    else:

        raise ValueError(f'Unknown shock_type. Please choose: parallel, key_rate or multi_tenor')