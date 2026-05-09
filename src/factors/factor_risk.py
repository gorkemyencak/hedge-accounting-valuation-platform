import pandas as pd
import numpy as np
import scipy

class FactorRiskModel:
    """ 
    Factor Risk Model to map tenor-level DV01s into factor-level DV01s 

    Factor_DV01 Formula:
        Factor_DV01 = V^-1 x DV01_tenor

        where:
            V -> eigenvectors (factor loadings)
            and 
            DV01_tenor -> risk per tenor
    
    PnL Formula:
        PnL = Factor_DV01^-1 x z

        where:
            Factor_DV01 -> risk per factor
            and 
            z -> factor returns
    """

    def __init__(
            self,
            pca_model
    ):
        """ 
        pca_model: YieldCurvePCA
            Fitted PCA model containing:
                - eigenvectors
                - eigenvalues
                - factor_returns 
        """
        self.pca_model = pca_model
        self.V = pca_model.eigenvectors
        self.factor_returns = pca_model.factor_returns
    

    def tenor_to_factor_exposure(
            self,
            dv01_vector
    ):
        """
        Projecting key-rate DV01 into PCA factor space

        Formula:
            Factor_DV01 = V^-1 x DV01_tenor 
                ->  (self.n_factors x 1) = (tenor x self.n_factors)^T x (tenor x 1)
        """
        # reshaping (tenor, ) dv01 series to (tenor x 1) dv01 vector
        dv01 = dv01_vector.values.reshape(-1, 1)

        # factor_dv01 vector -> (self.n_factors x 1)
        factor_dv01 = self.V.T @ dv01

        # flattening (self.n_factors x 1) dv01 vector into (self.n_factors, ) dv01 series
        factor_dv01 = factor_dv01.flatten()

        return pd.Series(
            factor_dv01,
            index = [f'Factor_{i}' for i in range(self.pca_model.n_factors)]
        )
    

    def portfolio_factor_exposure(
            self,
            dv01_dict
    ):
        """
        Aggregating factor exposures across multiple trades in a swap portfolio

        dv01_dict = {
                'swap_A': dv01_series,
                'Swap_B': dv01_series
        }

        Formula:
            Factor_DV01_portfolio = V^-1 x Portfolio_DV01
                -> (self.n_factors x 1) = (tenor x self.n_factors)^T x (tenor x 1)

            where:
                Portfolio_DV01 = sum(DV01_trade)
        """
        # aggregating across swaps -> (tenor, ) dv01 series
        total_dv01 = sum(dv01_dict.values())

        return self.tenor_to_factor_exposure(dv01_vector = total_dv01)
    

    def factor_covariance(self):
        """ Covariance matrix of factor returns """
        return self.factor_returns.cov()
    

    def factor_var(
            self,
            factor_exposure,
            confidence = 0.99
    ):
        """ 
        Compute parametric VaR using factor model 
        
        Portfolio PnL Formula:
            PnL = f^T x z

            where:
                f -> factor exposures -> (self.n_factors x 1)
                z -> factor returns -> (1 x self.n_factors)

        Variance Formula:
            Var(PnL) = f^T x Cov(z) x f -> (1 x 1) = (1 x self.n_factors) x (self.n_factors x self.n_factors) x (self.n_factors x 1)

        VaR Formula:
            VaR = z_alpha x sqrt(f^T x Cov(z) x f)

            where:
                sqrt(f^T x Cov(z) x f) -> Vol(PnL)
        """
        # reshaping (self.n_factors, ) factor exposures series to (self.n_factors x 1) factor exposures vector
        f = factor_exposure.reshape(-1, 1)
        
        # covariance matrix
        Sigma = self.factor_covariance().values

        # PnL Variance
        pnl_variance = f.T @ Sigma @ f

        # PnL Volatility
        pnL_vol = np.sqrt(pnl_variance)[0, 0]

        # VaR
        VaR = scipy.stats.norm.ppf(confidence) * pnL_vol

        return float(VaR)
    

    def historical_var(
            self,
            factor_exposure,
            confidence
    ):
        """
        Historical VaR using factor return time-series

        Historical PnL Formula 
            Historical PnL = z_t x f -> (dates x 1) = (dates x self.n_factors) x (self.n_factors x 1)

            where:
                f -> factor exposures -> (self.n_factors x 1)
                z_t -> factor returns -> (dates x self.n_factors)

        VaR: quantile of historical PnL distribution
        """
        # (self.n_factors, ) factor exposures serie
        f = factor_exposure.values
        
        # factor returns time-series -> (dates x self.n_factors)
        factor_moves = self.factor_returns.values

        # PnL time series -> (dates x 1)
        historical_pnl = factor_moves @ f

        # the quantile of PnL distribution
        VaR = -np.quantile(historical_pnl, 1 - confidence)

        return float(VaR)
    

    def shock_portfolio(
            self,
            factor_exposure,
            level_shock_bp = 1.0,
            slope_shock_bp = 1.0,
            curvature_shock_bp = 1.0
    ):
        """
        Stress testing with factor shocks

        Shock Vector Formula (in rate):
            s = [ΔLevel, ΔSlope, ΔCurvature]
        
        PnL Shock Formula:
            PnL_shock = f^T x s -> (1 x 1) = (1 x self.n_factors) x (self.n_factors x 1)

            where:
                f -> factor exposures -> (self.n_factors x 1)
                s -> factor shocks -> (self.n_factors x 1)
        """
        # reshaping (self.n_factors, ) factor exposures serie to (self.n_factors, 1) factor exposures vector
        f = factor_exposure.values.reshape(-1, 1)

        # shock vecctor
        shock_vector = np.array([
             [level_shock_bp/10000],
             [slope_shock_bp / 10000],
             [curvature_shock_bp / 10000]
        ])

        # pnl
        pnl = f.T @ shock_vector

        return float(pnl[0][0])
    

    def factor_to_yield_shock(
            self,
            factor_shock
    ):
        """ 
        Convert factor shocks into full yield curve move 
        
        Formula:
            Δy = V x z_shock -> (tenor x 1) = (tenor x self.n_factors) x (self.n_factors x 1)

            where:
                V -> eigenvector -> (tenor x self.n_factors)
                z_shock -> factor shocks -> (self.n_factors x 1)
        """
        # yield shock
        yield_shock = self.V @ factor_shock

        return pd.Series(
            yield_shock.flatten(),
            index = self.pca_model.tenors
        )
    