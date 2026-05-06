import pandas as pd

from src.factors.yield_curve_pca import YieldCurvePCA
from src.factors.factor_risk import FactorRiskModel

class SwapFactorRiskPipeline:
    """
    Factor risk engine for swap portfolios

    Financial flow of the pipeline:
        Swap Portfolio -> Key-rate DV01 -> PCA -> Fator Risk -> VaR / Stress Testing

    Outputs:
        - Level/slope/Curvature factor risk exposures
        - Parametric VaR
        - Historical VaR
        - Stress testing
    """
    def __init__(
            self,
            swap_portfolio,
            zero_curve,         # to be used for PCA
            discount_curve,     # to be used for pricing & DV01 computation
            key_rate_tenors
    ):
        # attributes
        self.portfolio = swap_portfolio
        self.zero_curve = zero_curve
        self.discount_curve = discount_curve
        self.key_rate_tenors = key_rate_tenors

    # Step 1 - Fitting PCA model
    def fit_pca(self):
        """ Extract yield curve factors from history """
        self.pca_model = YieldCurvePCA(n_factors = 3)
        self.pca_model.fit(zero_curve = self.zero_curve)

        self.factor_model = FactorRiskModel(pca_model = self.pca_model)

    # Step 2 - Computing portfolio key-rate DV01
    def compute_portfolio_key_rate_dv01(self):
        """ DV01 across key tenors (Sensitivity of portfolio value to key tenors) """
        # portfolio-level DV01
        dv01 = self.portfolio.portfolio_dv01(
            df_curve = self.discount_curve,
            shock_type = 'key_rate',
            key_rate_tenors = self.key_rate_tenors
        )

        # use most recent DV01s
        most_recent_dv01 = dv01.iloc[-1]
        most_recent_dv01.index = [float(c.split('_')[-1][:-1]) for c in dv01.columns]

        # attribute assignment
        self.portfolio_dv01 = most_recent_dv01

        return most_recent_dv01
    
    # Step 3 - Convert DV01 -> Factor exposure
    def compute_factor_exposure(self):
        """ Translating key-rate DV01 into LEvel/slope/Curvature exposure """
        # projecting key-rate DV01 into PCA factor space
        factor_exposure = self.factor_model.tenor_to_factor_exposure(
            dv01_vector = self.portfolio_dv01
        )

        # attribute assignment
        self.factor_exposure = factor_exposure

        return factor_exposure
    
    # Step 4 - Parametric VaR
    def compute_parametric_var(self, confidence = 0.99):
        """ Computing factor covariance VaR using factor model """
        return self.factor_model.factor_var(
            factor_exposure = self.factor_exposure,
            confidence = confidence
        )
    
    # Step 5 - Historical VaR
    def compute_historical_var(self, confidence = 0.99):
        """ Computing historical simulation VaR using factor returns """
        return self.factor_model.historical_var(
            factor_exposure = self.factor_exposure,
            confidence = confidence
        )
    
    # Step 6 - Factor stress testing
    def stress_test(
            self,
            level_bp = 1.0,
            slope_bp = 1.0,
            curvature_bp = 1.0
    ):
        """ Stress portfolio using factor shocks given in basis points """
        # PnL
        pnl = self.factor_model.shock_portfolio(
            factor_exposure = self.factor_exposure,
            level_shock_bp = level_bp,
            slope_shock_bp = slope_bp,
            curvature_shock_bp = curvature_bp
        )

        # factor shocks
        factor_shocks = pd.Series({
            'Factor_1': level_bp / 10000,
            'Factor_2': slope_bp / 10000,
            'Factor_3': curvature_bp / 10000
        })

        # reconstructing full yield curve move
        yield_shock = self.factor_model.factor_to_yield_shock(
            factor_shock = factor_shocks
        )

        return pnl, factor_shocks
    
    # Step 7 - Risk Report
    def risk_report(self, confidence = 0.99):
        """ Running the entire factor risk pipeline and returning the full summary risk report """
        # Step 1
        self.fit_pca()

        # Step 2
        dv01 = self.compute_portfolio_key_rate_dv01()

        # Step 3
        factor_exposure = self.compute_factor_exposure()

        # Step 4
        parametric_var = self.compute_parametric_var(confidence = confidence)

        # Step 5
        historical_var = self.compute_historical_var(confidence = confidence)

        risk_report = pd.DataFrame({
            'Metric': [
                'Level Exposure',
                'Slope Exposure',
                'Curvature Exposure',
                f'Parametric VaR ({str(confidence)}%)',
                f'Historical VaR ({str(confidence)}%)'
            ],
            'Value': [
                factor_exposure.iloc[0],
                factor_exposure.iloc[1],
                factor_exposure.iloc[2],
                parametric_var,
                historical_var
            ]
        })

        return risk_report
