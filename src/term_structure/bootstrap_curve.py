import pandas as pd

from src.term_structure.conventions import Tenors_To_Yearfrac, Treasury_Tenors_To_Years, Coupon_Freq
from src.term_structure.discount_factor_conversions import df_from_simple_rate, zero_rate_from_df
from src.term_structure.bond_pricing import get_coupon_times, bond_pricing_from_df, solve_last_df, solve_df_from_par_bond
from src.term_structure.curve_interpolation import build_coupon_structure


def bootstrap_df_from_sofr(sofr_curve: pd.DataFrame) -> pd.DataFrame:
    """ Bootstrap discount factors from money market quotes (SOFR curve) """
    discount_curves = []

    for date, row in sofr_curve.iterrows():
        dfs = {}

        for tenor, rate in row.items():
            if tenor not in Tenors_To_Yearfrac:
                continue
            
            yearfrac = Tenors_To_Yearfrac[str(tenor)]

            if pd.isna(rate):
                dfs[tenor] = None
                continue

            df = df_from_simple_rate(
                rate = rate,
                yearfrac = yearfrac
            )

            dfs[tenor] = df
        
        discount_curves.append(pd.Series(
            dfs,
            name = date
        ))
    
    return pd.DataFrame(discount_curves)


def bootstrap_df_from_treasury_curve(
        treasury_curve: pd.DataFrame, # par-yields
        interpolated_dfs: pd.DataFrame, # short-end DFs
        freq = 2,
        face = 100 
):
    """ 
    Bootstrap discount factors from Treasury par yields 
    
    It merges SOFR discount curve (short-end) and treasury par yields (long-end)
    """
    # building coupon times structure
    coupon_times = build_coupon_structure(
        max_year = 30,
        freq = 2
    )

    bootstrapped_curves = []

    for date in treasury_curve.index:

        treasury_row = treasury_curve.loc[date]

        # Starting with building short-end of the discount curve from interpolated money market dfs
        df_interp_row = interpolated_dfs.loc[date]

        # building known short-end DFs
        known_dfs = interpolated_dfs.loc[date].to_dict()

        for tenor, maturity in Treasury_Tenors_To_Years.items():

            y = treasury_row[tenor] / 100
            coupon = y * face / freq

            times = coupon_times[coupon_times <= maturity]

            pv_known = 0.0
            for t in times[:-1]:
                pv_known += coupon * known_dfs[t]
            
            df_T = solve_df_from_par_bond(
                pv_known = pv_known,
                coupon = coupon,
                face = face,
                price = 100.0
            )

            known_dfs[maturity] = df_T
        
        bootstrapped_curves.append(
            pd.Series(
                known_dfs,
                name = date
            )
        )

    return pd.DataFrame(bootstrapped_curves).sort_index(axis=1)
