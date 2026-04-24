import numpy as np
import pandas as pd

from src.term_structure.conventions import Tenors_To_Yearfrac

from src.term_structure.discount_factor_conversions import zero_rate_from_df

# zero-curve builder from SOFR
def build_zero_curve_from_sofr_dfs(discount_curve: pd.DataFrame) -> pd.DataFrame:
    """ Converting money market discount factors into zero rates """
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


# generic zero-curve builder
def build_zero_curve_from_discount_curve(
        discount_curve: pd.DataFrame
) -> pd.DataFrame:
    """  
    Converting bootstrapped discount factors into continuously compounded zero curve

    Formula:
        Z(t) = -ln(DF(t)) / t
    """
    zero_curves = []

    for date, row in discount_curve.iterrows():
        
        zero_rates = {}

        for maturity, df in row.items():

            if maturity == 0:
                continue

            zero_rate = -np.log(df) / maturity
            zero_rates[maturity] = 100 * zero_rate

        zero_curves.append(
            pd.Series(
                zero_rates,
                name = date
            )
        )

    return pd.DataFrame(zero_curves)