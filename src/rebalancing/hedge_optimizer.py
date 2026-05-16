import numpy as np
import pandas as pd

from src.hedging.hedge_ratio import HedgeRatio

class HedgeOptimizer:
    """ 
    Hedge Optimizer module that computes hedge notionals using the full PCA hedge construction pipeline.

    Pipeline at rebalance date t:
        i) Compute portfolio key-rate DV01
        ii) Convert to factor exposure
        iii) Compute hedge instruments key-rate DV01
        iv) Convert to factor matrix
        v) Solve hedge notionals
    """
    def __init__(
            self,
            factor_risk_model,
            key_rate_tenors: list[float]
    ):
        """
        Parameters:
            factor_risk_model: FactorRiskModel
            key_rate_tenors: list of key-rate tenors        
        """
        self.factor_risk_model = factor_risk_model
        self.key_rate_tenors = key_rate_tenors
    

    # portfolio factor exposure
    def compute_portfolio_factor_exposure(
            self,
            portfolio,
            df_curve_history,
            rebalance_date
    ):
        """ Factor exposure of portfolio at rebalance date """
        # use curve history up to rebalance date
        curve_r = df_curve_history.loc[:rebalance_date]

        # portfolio key-rate DV01 timeseries
        dv01_ts = portfolio.portfolio_dv01(
            df_curve = curve_r,
            shock_type = 'key_rate',
            key_rate_tenors = self.key_rate_tenors
        )

        # get latest DV01 snapshot
        dv01_most_recent = dv01_ts.iloc[-1]

        # rename index into tenor labels for PCA model
        dv01_most_recent.index = self.factor_risk_model.pca_model.tenors

        # convert to factor exposure
        portfolio_factor_exposure = self.factor_risk_model.tenor_to_factor_exposure(
            dv01_vector = dv01_most_recent
        )

        return np.array(portfolio_factor_exposure)
    

    # hedge factor matrix
    def compute_hedge_factor_matrix(
            self,
            hedge_universe,
            df_curve_history,
            rebalance_date
    ):
        """ Builds hedge factor matrix f_h -> (nb_instruments x self.n_factors) """
        # use curve history up to rebalance date
        curve_r = df_curve_history.loc[:rebalance_date]

        # trade-level key-rate DV01
        trade_dv01_ts = hedge_universe.trade_dv01(
            df_curve = curve_r,
            shock_type = 'key_rate',
            key_rate_tenors = self.key_rate_tenors
        )

        # get latest DV01 snapshot
        dv01_most_recent = trade_dv01_ts.iloc[-1]

        # reshaping dv01_most_recent into (nb_instruments x tenor) matrix
        nb_instruments = len(hedge_universe.instruments)
        nb_tenors = len(self.key_rate_tenors)

        dv01_matrix = dv01_most_recent.values.reshape(nb_instruments, nb_tenors)

        hedge_factor_vectors = []

        for i in range(nb_instruments):

            # initialize full key-rate DV01 vector
            dv01_vector = pd.Series(
                dv01_matrix[i],
                index = self.key_rate_tenors
            )

            # convert to factor exposure
            factor_exposure = self.factor_risk_model.tenor_to_factor_exposure(
                dv01_vector = dv01_vector
            )

            hedge_factor_vectors.append(np.array(factor_exposure))
        
        f_h = np.vstack(hedge_factor_vectors)

        return f_h
    

    # solve for optimal hedge notionals
    def solve_opt_hedge(
            self,
            portfolio,
            hedge_universe,
            df_curve,
            rebalance_date
    ):
        """ 
        Solving for optimal hedge notionals 
        
        Returns optimal hedge notionals and residual factor exposures
        """
        # compute portfolio factor exposure
        f_p = self.compute_portfolio_factor_exposure(
            portfolio = portfolio,
            df_curve_history = df_curve,
            rebalance_date = rebalance_date
        )

        # compute hedge factor matrix
        f_h = self.compute_hedge_factor_matrix(
            hedge_universe = hedge_universe,
            df_curve_history = df_curve,
            rebalance_date = rebalance_date
        )

        hedge_notionals = HedgeRatio.solve_hedge_notionals(
            portfolio_factor_exposure = f_p,
            hedge_factor_matrix = f_h
        )

        residual_factor_exposure = HedgeRatio.residual_factor_exposure(
            portfolio_factor_exposure = f_p,
            hedge_factor_matrix = f_h,
            hedge_notionals = hedge_notionals
        )

        return hedge_notionals, residual_factor_exposure
    