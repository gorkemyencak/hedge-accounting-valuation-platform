import pandas as pd
import numpy as np

class PnLEngine:
    """ Converts factor exposures and factor returns into daily portfolio PnL time series """

    @staticmethod
    def factor_pnl_timeseries(
        factor_returns,
        factor_exposures,
    ):
        """
        Computing daily PnL from factor model

        PnL Formula:
            PnL_t = Z x f   -> (dates x 1) = (dates x self.n_factors) x (self.n_factors x 1)

            where:
                Z -> (PCA) factor returns time-series -> (dates x self.n_factors)
                f -> portfolio factor exposures -> (self.n_factors x 1)
        """
        # factor returns and factor exposures
        Z = factor_returns.values               # (dates x self.n_factors)
        f = np.array(factor_exposures).reshape(-1, 1)     # (self.nfactors x 1)

        # computing pnl time-series
        pnl = Z @ f

        pnl = pnl.flatten()

        return pd.Series(
            pnl,
            index = factor_returns.index,
            name = 'Daily_PnL'
        )
    

    @staticmethod
    def cumulative_pnl(
        pnl_timeseries
    ):
        """ Cumulative PnL time-series """
        return pnl_timeseries.cumsum().rename('Cumulative_PnL')
    
    @staticmethod
    def pnl_volatility(
        pnl_timeseries
    ):
        """ 
        Compute Daily PnL Volatility 
        
        Formula:
            sigma = std(PnL)
        """
        return pnl_timeseries.std()
    