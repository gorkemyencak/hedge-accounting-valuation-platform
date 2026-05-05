import pandas as pd
import numpy as np

class YieldCurvePCA:
    """
    Principal Component Analysis (PCA) for yield curve risk factors

    Daily yield curve changes are modeled with respect to:
        ΔZ(t, Tenor) = a_1(t)F_1(Tenor) + a_2(t)F_2(Tenor) + a_3(t)F_3(Tenor) + error_term

        where:
            F_1 -> Level factor
            F_2 -> Slope factor
            F_3 -> Curvature factor
        and
            a_i(t) -> factor returns for i = 1, .., 3
        
        
    PCA model allows to extract above-mentioned factors from historical yield changes

    We will use rate changes as they are stationary time-series

    Model Inputs:
        - zero_curve
    
    Model Outputs:
        - eigenvectors -> factor loadings across maturities
        - eigenvalues -> variance explained
        - factor_returns -> daily factor time-series    
    """

    def __init__(
            self,
            n_factors = 3 # Level, Slope, Curvature
    ):
        
        self.n_factors = n_factors

        self.tenors = None
        self.delta_yields = None
        self.mean_changes = None
        self.cov_matrix = None
        self.eigenvalues = None
        self.eigenvectors = None
        self.factor_returns = None


    # helper function to get most important PCAs
    @staticmethod
    def get_pcs(eigen_vecs):
        """
        Returns first 3 principal components in financial order

        np.linalg.eigh returs eigenvalues in ascending order

        Therefore,
            last column -> PC1 (Level)
            second last -> PC2 (Slope)
            third last -> PC3 (Curvature)
        """
        pc1 = eigen_vecs[:, -1]
        pc2 = eigen_vecs[:, -2]
        pc3 = eigen_vecs[:, -3]

        return pc1, pc2, pc3
    
    
    def fit(
            self, 
            zero_curve: pd.DataFrame
    ):
        """ 
        Computing PCA by fitting in the historical yield curve 
        
        Steps to be followed:
            - Selecting liquid market tenors
            - Compute daily yield changes ΔZ
            - Center the data
            - Build covriance matrix
            - Eigen decomposition
            - Compute factor return time-series
        """
        # selecting liquid market tenors
        key_tenors = [1, 2, 3, 5, 7, 10, 20, 30]
        zero_curve = zero_curve[key_tenors]

        # storing tenor labels
        self.tenors = zero_curve.columns.astype(float)

        # computing daily yield changes
        delta_yields = zero_curve.diff().dropna()
        self.delta_yields = delta_yields

        # centering data by subtracting mean changes per tenor
        self.mean_changes = self.delta_yields.mean()
        centered_delta_yields = self.delta_yields - self.mean_changes

        # building covariance matrix
        self.cov_matrix = centered_delta_yields.cov()

        # eigen decomposition
        eigen_vals, eigen_vecs = np.linalg.eigh(self.cov_matrix)

        # sorting eigen values and vectors in descending order -> largest variance comes first in the order
        idx = np.argsort(eigen_vals)[::-1]
        eigen_vals = eigen_vals[idx]
        eigen_vecs = eigen_vecs[:, idx]

        # keeping only first n factors that are already sorted in descending order
        self.eigenvalues = eigen_vals[:self.n_factors]              # returns 1 x self.n_factors array
        self.eigenvectors = eigen_vecs[:, :self.n_factors]          # returns key_tenors x self.n_factors array

        # factor returns (projection) -> time-series of factor moves of level/slope/curvature factor returns
        factor_returns = np.dot(centered_delta_yields.values, self.eigenvectors)

        self.factor_returns = pd.DataFrame(
            factor_returns,
            index = centered_delta_yields.index,
            columns = [f'Factor_{i}' for i in range(self.n_factors)]
        )

        return self


    def get_factor_loadings(self) -> pd.DataFrame:
        """ 
        Returns factor loading curves -> F_i(Tenor) for i = 1, .., self.n_factors
            -> Level/Slope/Curvature shapes, respectively        
        """
        loadings = pd.DataFrame(
            self.eigenvectors,
            index = self.tenors,
            columns = [f'Factor_{i}' for i in range(self.n_factors)]
        )

        return loadings
    

    def get_factor_returns(self):
        """ Daily time-series of factor moves """
        return self.factor_returns
    

    def explained_variance(self):
        """ 
        Explained Variance table 
            -> percentage of yield curve variance explained by each factor
        """
        total_variance = self.eigenvalues.sum() # type: ignore
        explained_var = self.eigenvalues / total_variance

        return pd.DataFrame({
            'EigenValue': self.eigenvalues,
            'Explained_Variance': explained_var,
            'Cumulative_Explained': np.cumsum(explained_var)
        }, index = [f'Factor_{i}' for i in range(self.n_factors)]
        )


    def reconstruct_curve(self):
        """ 
        Validation tool by reconstructing yield changes using PCA factors and factor loadings 
        
        Formula:
            ΔZ = F_i * factor_loadings^-1   -> (date x self.n_factors) x (self.tenors x self.n_factors)^T -> (date x self.tenors)
        """
        reconstructed_curve = np.dot(self.factor_returns.values, self.eigenvectors.T)  # type: ignore
        reconstructed_curve = pd.DataFrame(
            reconstructed_curve,
            index = self.factor_returns.index, # type: ignore
            columns = self.tenors
        ) 
