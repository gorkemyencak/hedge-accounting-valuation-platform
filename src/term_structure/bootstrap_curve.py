import pandas as pd

from src.term_structure.conventions import Tenors_To_Yearfrac
from src.term_structure.discount_factor_conversions import df_from_simple_rate, zero_rate_from_df

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


def build_zero_curve_from_df(discount_curve: pd.DataFrame) -> pd.DataFrame:
    """ Converting discount factors into zero rates """
    zero_curves = []

    for date, row in discount_curve.iterrows():
        zero_rates = {}

        for tenor, df in row.items():
            if tenor not in Tenors_To_Yearfrac:
                continue
            
            yearfrac = Tenors_To_Yearfrac[str(tenor)]

            if pd.isna(df):
                zero_rates[tenor] = None
                continue

            zero_rate = 100 * zero_rate_from_df(
                df = df,
                yearfrac = yearfrac
            )

            zero_rates[tenor] = zero_rate
        
        zero_curves.append(
            pd.Series(
                zero_rates,
                name = date
            )
        )
    
    return pd.DataFrame(zero_curves)
