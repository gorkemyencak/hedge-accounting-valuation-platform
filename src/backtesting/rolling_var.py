import pandas as pd
import numpy as np
import scipy

from src.backtesting.pnl_engine import PnLEngine

class RollingVaR:
    """ Computes rolling VaR time-series using FactorRiskModel """
    def __init__(
            self,
            factor_risk_model
    ):
        # attributes
        self.frm = factor_risk_model
        self.factor_returns = factor_risk_model.factor_returns


    def rolling_historical_var(
            self,
            exposures,
            window: int = 252,
            alpha: float = 0.99
    ):
        """ 
        Compute rolling VaR time-series 
        
        Returns forecasted VaR_t(T-window) for each day t

        Formula:
            VaR_t = -quantile( PnL_t{t-window:t}, 1 - confidence )
        
            where:
                PnL_t: PnL time-series (dates x 1)
                window: rolling window length
                alpha: confidence level
        """
        # compute PnL time-series from factor exposures
        pnl_series = PnLEngine.factor_pnl_timeseries(
            factor_returns = self.factor_returns,
            factor_exposures = exposures
        )

        var_list = []
        dates = []

        for t in range(window, len(pnl_series)):

            pnl_window = pnl_series.iloc[t-window:t]

            var_t = -np.quantile(pnl_window, 1 - alpha)

            var_list.append(var_t)
            dates.append(pnl_series.index[t])
        
        return pd.Series(
            var_list,
            index = dates,
            name = 'Historical_VaR'
        )
    

    def rolling_factor_var(
            self,
            exposures,
            window: int = 252,
            alpha: float = 0.99
    ):
        """
        Compute rolling parametric (variance-covariance) VaR time-series

        Formula:
            VaR_t = z_alpha * sqrt(f^T x Cov(z_t) x f)

            where:
                Cov(z_t) -> Cov(z_{t-window:t})
        """
        # factor returns and factor exposures
        z = self.factor_returns                 # (dates x self.n_factors)
        f = np.array(exposures).reshape(-1, 1)  # (self.n_factors x 1)

        # z_alpha for a given confidence level
        z_alpha = scipy.stats.norm.ppf(alpha)

        var_list = []
        dates = []

        for t in range(window, len(self.factor_returns)):

            window_returns = z.iloc[t-window:t]

            # rolling covariance
            sigma_t = window_returns.cov().values

            # variance (f^T x sigma_t x f)
            pnl_var = f.T @ sigma_t @ f
            pnl_vol = np.sqrt(pnl_var)[0, 0]

            # compute VaR_t
            var_t = z_alpha * pnl_vol

            var_list.append(var_t)
            dates.append(z.index[t])
        
        return pd.Series(
            var_list,
            index = dates,
            name = 'Factor_VaR'
        )
    