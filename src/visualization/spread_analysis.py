import pandas as pd
import numpy as np

class SpreadAnalysis:
    """ Spread time-series analysis between two curves """
    def __init__(
            self,
            data: pd.DataFrame
    ):
        
        self.df = data.copy()

    # helper method for multi-index columns
    def _get_series(
            self,
            curve: str,
            tenor: str
    ) -> pd.Series:
        
        return self.df[(curve, tenor)]
    
    # computing the spread between treasury and sofr curves
    def compute_teasury_sofr_spread(self):

        treasury_10Y = self._get_series(
            curve = 'treasury',
            tenor = '10Y'
        )

        soft_ON = self._get_series(
            curve = 'sofr',
            tenor = 'ON'
        )

        spread = treasury_10Y - soft_ON
        spread.name = 'Treasury_SOFR_Spread'

        return spread
    
    # rolling volatility of spread
    def rolling_volatility(
            self,
            window: int = 90
    ) -> pd.Series:
        
        spread = self.compute_teasury_sofr_spread()
        vol = spread.rolling(window).std() * np.sqrt(252)
        vol.name = 'Spread_Rolling_Volatility'

        return vol
    
    
        


    
