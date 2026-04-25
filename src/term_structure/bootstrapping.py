import pandas as pd
import numpy as np

from scipy.optimize import brentq

from src.term_structure.conventions import Tenors_To_Yearfrac, Treasury_Tenors_To_Years
from src.term_structure.conversions import simple_rate_to_df


def build_coupon_structure(
        max_year = 30,
        freq = 2 
):
    """ Building coupon grid along the maturity based on coupon payment frequency """
    step = 1 / freq
    return np.arange(step, max_year + step, step)


# short-end bootstrapping
def bootstrap_dfs_from_sofr(
        sofr_curve: pd.DataFrame
) -> pd.DataFrame:
    """ Bootstrap discount factors from money market quotes (SOFR curve) """
    discount_curves = []

    for date, row in sofr_curve.iterrows():

        dfs = {}

        for tenor, rate in row.items():

            if tenor not in Tenors_To_Yearfrac:
                continue
            
            yearfrac = Tenors_To_Yearfrac[str(tenor)]

            if pd.isna(rate):
                continue

            dfs[yearfrac] = simple_rate_to_df(
                rate = rate/100,
                t = yearfrac
            )
        
        discount_curves.append(
            pd.Series(
                dfs,
                name = date
            )
        )

    return pd.DataFrame(discount_curves)


# long-end bootstrapping
def bootstrap_dfs_from_treasury(
        treasury_curve: pd.DataFrame,
        short_dfs: pd.DataFrame,
        freq: int = 2,
        face: float = 100.0,
        price: float = 100.0
) -> pd.DataFrame:

    """
    Bootstrap discount factors at treasury maturities 

    Formula:
        Bond pricing:
            100 = sum(C * DF(t_i)) + (100 + C) * DF(T)

                -> price = pv_known + (face + coupon) * DF(T)
        
        and
        Unknown DF for a given t, let t_last be the last known df:
            DF(t) = DF(t_last) * ( DF(T) / DF(t_last) )^((t - t_last)/(T - t_last))
    """
    discount_curves = []

    for date in treasury_curve.index:
        
        known_dfs = short_dfs.loc[date].dropna().to_dict()        
        treasury_row = treasury_curve.loc[date]

        for tenor, maturity in Treasury_Tenors_To_Years.items():

            y = treasury_row[tenor] / 100   # coupon rate
            coupon = face * y / freq    # coupon payment

            times = [i/freq for i in range(1, int(maturity * freq) + 1)]

            # splitting known & unknown times
            known_times = [t for t in times if t in known_dfs]
            unknown_times = [t for t in times if t not in known_dfs]

            # pv of known coupons
            pv_known = sum(coupon * known_dfs[t] for t in known_times)

            t_last = max(known_times)
            df_last = known_dfs[t_last]
            T = max(times)

            # function to solve DF(T)
            def bond_price_equation(df_T):

                pv_unknown = 0.0

                # iterating over coupons before maturity -> let t_last be the last known df
                for t in unknown_times[:-1]: 
                    w = (t - t_last) / (T - t_last)

                    df_t = df_last * (df_T / df_last) ** w
                    pv_unknown += coupon * df_t

                pv_final = (face + coupon) * df_T

                return pv_known + pv_unknown + pv_final - price

            # solve DF(T)
            df_T = brentq(
                f = bond_price_equation,
                a = 0.0,
                b = 1.0
            )

            # filling all missing coupon dfs
            for t in unknown_times:
                
                w = (t - t_last) / (T - t_last)
                known_dfs[t] = df_last * (df_T / df_last) ** w 


        discount_curves.append(
            pd.Series(
                known_dfs,
                name = date
            )
        )    

    return pd.DataFrame(discount_curves)

