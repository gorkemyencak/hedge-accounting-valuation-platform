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
    DV01 = NPV_shocked - NPV_base
"""

def swap_dv01(
        df_curve:pd.DataFrame,
        maturity: float,
        fixed_rate: float,
        freq: int = 2,
        notional: float = 1_000_000,
        shock_bps: float = 1.0
):
    """ Computing DV01 of a payer swap (pay-fixed, recieve float) using parallel shock """
    # Base NPV
    npv_base = swap_npv(
        df_curve = df_curve,
        maturity = maturity,
        fixed_rate = fixed_rate,
        freq = freq,
        notional = notional
    )

    # DF to Zero curve
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

    # Compute DV01
    dv01 = npv_shocked - npv_base

    return dv01
