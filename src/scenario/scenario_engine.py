import pandas as pd
import numpy as np

class ScenarioEngine:
    """
    Forward-looking scenario generator using Factor Risk Model

    To generate custom shock scenarios, PCA loadings will be utilized to convert yield curve shocks
    in bps into PCA factor shocks

    Factor Risk Model Inputs:
        factor_returns (dates x self.n_factors)
        factor_cov     (self.n_factors x self.n_factors)
    """
    def __init__(self, factor_risk_model):
        
        self.factor_risk_model = factor_risk_model
        self.factor_returns = self.factor_risk_model.factor_returns
        self.factor_cov = self.factor_risk_model.factor_covariance()
        self.n_factors = self.factor_returns.shape[1]
        self.pca_loadings = self.factor_risk_model.pca_model.eigenvectors 
        self.tenors = self.factor_risk_model.pca_model.tenors

    # helper methods
    def yield_to_factor_shock(
            self,
            yield_shock_vector
    ):
        """ 
        Convert yield curve shock (bps) vector into PCA factor shock vector

        Formula:
            Δy = V x z_shock -> (tenor x 1) = (tenor x self.n_factors) x (self.n_factors x 1)

            where:
                V -> eigenvector -> (tenor x self.n_factors)
                z_shock -> factor shocks -> (self.n_factors x 1)

            To solve for z_shock, apply Moore-Pensore pseudo-inverse:
                z_shock = (V^T x V)^-1 x V^T x Δy -> (self.n_factors x 1) = (self.n_factors x self.n_factors) x (self.n_factors x tenor) x (tenor x 1)
        """
        # eigenvector
        V = self.pca_loadings.values    # (tenor x self.n_factors)

        # yield shock
        dy = np.array(yield_shock_vector).reshape(-1, 1)    # (tenor x 1)

        # Moore-Penrose pseudo-inverse
        V_pinv = np.linalg.pinv(V)      # (self.n_factors x tenor)

        # compute factor shock vector
        z_shock = V_pinv @ dy       # (self.n_factors x 1)

        return z_shock.flatten()


    # historical scenario generator
    def historical_scenarios(
            self, 
            n_scenarios: int = 100
    ):
        """
        Select worst historical factor shocks
        
        Returning a factor return matrix for n_scenarios (n_scenarios x self.n_factors)
        """
        # computing the magnitude of factor moves
        shock_magnitude = np.linalg.norm(
            self.factor_returns,
            axis = 1
        )

        # selecting historical shocks with largest magnitude of factor moves
        worst_case_index = np.argsort(shock_magnitude)[-n_scenarios:]

        Z_hist = self.factor_returns.iloc[worst_case_index]

        return Z_hist.reset_index(drop = True)

    
    # Monte Carlo scenario generator
    def monte_carlo_scenarios(
            self,
            n_scenarios: int = 100,
            random_seed: int = 7
    ):
        """
        Generate Monte Carlo factor shocks using Gaussian Distribution with z ~ N(0, Sigma)

        Returning a factor return matrix for n_scenarios (n_scenarios x self.n_factors)
        """
        # setting random seed
        np.random.seed(seed = random_seed)

        # MC scenarios using multi-variate Gaussian distribution
        Z_mc = np.random.multivariate_normal(
            mean = np.zeros(self.n_factors),
            cov = self.factor_cov,
            size = n_scenarios
        )

        return pd.DataFrame(
            Z_mc,
            columns = self.factor_returns.columns
        )
    
    # user-defined custom shock generator
    def custom_scenarios(
              self,
              parallel_shock_bps: int = +100,
              short_rate_bps: int = +150,
              steepener_bps: int = +60,
              flattener_bps: int = +60
    ):
        """
        Deterministic yield curve stress scenarios defined in bps and converted into PCA space

        Returning a factor return matrix (tenor x self.n_factors)
        """
        # nb of tenors
        N = len(self.tenors)

        # rate shock adjustments
        parallel_shock = +100 if parallel_shock_bps == 0 else abs(parallel_shock_bps)
        short_rate_shock = +150 if abs(short_rate_bps) < 10 else abs(short_rate_bps) 
        steepener_shock = +60 if abs(steepener_bps) < 10 else abs(steepener_bps)
        flattener_shock = +60 if abs(flattener_bps) < 10 else abs(flattener_bps)
 
        # yield curve shock scenarios
        yc_shock_scenarios = {
             # 100 bps paralel shift scenarios
             'Parallel Up': np.ones(N) * parallel_shock,
             'Parallel Down': np.ones(N) * -parallel_shock,

             # short rate up
             'Short Up': np.linspace(short_rate_shock, +10, N),
            
             # short rate down
             'Short Down': np.linspace(-short_rate_shock, -10, N),

             # curve steepener
             'Steepener': np.concatenate([
                  np.ones(N//2) * np.linspace(-steepener_shock, -10, N//2),
                  np.ones(N - N//2) * np.linspace(+10, steepener_shock, N - N//2)
             ]),

             # curve flattener
             'Flattener': np.concatenate([
                  np.ones(N//2) * np.linspace(flattener_shock, +10, N//2),
                  np.ones(N - N//2) * np.linspace(-10, -flattener_shock, N - N//2)
             ])
        }

        # computing factor scenarios
        factor_scenarios = {}

        for name, yield_shock in yc_shock_scenarios.items():
             
             factor_scenarios[name] = self.yield_to_factor_shock(
                  yield_shock_vector = yield_shock
             )

        Z_custom = pd.DataFrame.from_dict(
             factor_scenarios,
             orient = 'index',
             columns = self.factor_returns.columns
        )

        return Z_custom
