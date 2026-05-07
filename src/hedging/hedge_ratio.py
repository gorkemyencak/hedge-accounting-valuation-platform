import numpy as np
import pandas as pd

class HedgeRatio:
    """ 
    Solve for hedge notionals using PCA factor exposures 
    
    Hedging condition:
        f_p + F_h^T x w = 0
            -> w = (F_h^T)^-1 x -f_p

        where
            f_p -> portfolio_factor_exposure_vector -> (self.n_factors x 1)
            F_h -> hedge factor matrix -> (n_instruments x self.n_factors)
            w -> hedge notionals vector -> (n_instruments x 1)  
    """
    @staticmethod
    def solve_hedge_notionals(
        portfolio_factor_exposure,
        hedge_factor_matrix
    ):
        """ Returns hedge notionals vector using PCA factor exposures """
        # computing hegde notionals vector using Moore-Penrose pseudo inverse of a matrix
        w = np.linalg.pinv(hedge_factor_matrix.T) @ -portfolio_factor_exposure
    
        return w
    
    @staticmethod
    def residual_factor_exposure(
        portfolio_factor_exposure,
        hedge_factor_matrix,
        hedge_notionals
    ):
        """
        Residual Formula:
            residual = F_h^T x w + f_p
        """
        # residual computation
        res = hedge_factor_matrix.T @ hedge_notionals + portfolio_factor_exposure
        
        return res
    
    @staticmethod
    def hedge_report(
        hedge_names,
        hedge_notionals,
        portfolio_factor_exposure,
        residual_exposure
    ):
        """ Summary hedge report containing portfolio and residual exposures """
        hedge_df = pd.DataFrame({
            'HedgeInstrument': hedge_names,
            'Notional': hedge_notionals
        })

        exposure_df = pd.DataFrame({
            'PortfolioExposure': portfolio_factor_exposure,
            'ResidualExposure': residual_exposure
        }, index = [f'Factor_{i}' for i in range(len(portfolio_factor_exposure))]
        )

        return hedge_df, exposure_df
    