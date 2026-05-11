import pandas as pd
import numpy as np
from scipy.stats import chi2

class VaRBacktest:
    """
    VaR backtesting module implementing:
        - Exception counting
        - Kupiec unconditional coverage test
        - Basel exception classification

    This module verifies if VaR is correctly computed as a regulatory-level (Basel/FRTB) scope.

    Exception Indicator is going to tell whether a breach occurs in the vectorized for of PnL time-series
        - Breach condition -> PnL_t < -VaR_t
            -> this will tell us the number of breaches that PnL will exceed the worst-case scenario given a certain confidence level
        - Expected breach probability
            -> this method will also return the expected breach probability given a certain confidence level of VaR
    
    Kupiec Test (Likelihood Ratio)
        - Hypothesis testing to claim whether the observed breach probability matches the theoretical (expected) one
            -> Null Hypothesis: observed_breach_probability = expected_breach_probability
            -> Alternate Hypothesis: observed_breach_probability != expected_breach_probability
        - Computing the likelihood ratio statistics under null hypothesis 
            -> p_value to accept/reject the null hypothesis

    Basel exception classifier
        - Converting the number of observed breaches into regulatory zone
    """
    def __init__(
            self,
            confidence_level: float = 0.99
    ):
        # attributes
        self.alpha = confidence_level
        self.p = 1 - confidence_level   # expected breach probability
    

    # exception indicator series
    def compute_exception(
            self,
            pnl,
            var
    ):
        """
        Compute VaR exception indicator, returning (dates x 1) vector of 0/1 values denoting the breach time-series

        Breach condition:
            PnL_t < -VaR_t

            where:
                - PnL_t denotes the PnL at a given date t in PnL time-series 
                    -> PnL_t < 0, if loss, else otherwise
                - VaR_t denotes the maximum expected loss for a given confidence level at a given date t in daily VaR forecast time-series 
                    -> positive number
        
        Exception Indicator:
            sum(I_t)

            where:
                - I_t = 1 if PnL_t < -VaR_t, 
                        0 otherwise
        
        Observed Breach Probability:
            p^hat = N / T

            where:
                - N = sum(I_t)

        Expected Breach Probability (Theoretical)
            p = 1 - alpha

            where:
                - alpha: confidence level
        """
        return (pnl < -var).astype(int)
    
    # kupiec test hypothesis testing
    def kupiec_test(
            self,
            pnl,
            var
    ):
        """ 
        Kupiec unconditional coverage test 
        
        Hypothesis testing:
            - H_0: p^hat = p
              H_Alternate: p^hat != p
        
        Likelihood ratio statistic:
            LR = -2 x ln( ((1-p)^{T-N} x p^N) / ((1-p_hat)^{T-N} x p_hat^N) )

            -> under H_0:
                LR ~ chi2(1)
            
            and p_value = 1 - chi2.cdf(LR, df=1)

                -> reject H_0 if p_value < 0.05
        """
        # observed breach vector
        I = self.compute_exception(
            pnl = pnl,
            var = var
        )

        # date length
        T = len(I)
        
        # number of observed breaches
        N = I.sum()

        # observed breach probability
        p_hat = N / T

        # theoretical breach probability
        p = self.p

        # prohibiting log(0)
        eps = 1e-9
        p_hat = np.clip(p_hat, eps, 1 - eps)

       # compute likelihood ratio statistic 
        LR = -2 * np.log(
            ((1 - p) ** (T - N) * (p ** N))
            / ((1 - p_hat) ** (T - N) * (p_hat ** N))
        )

        # computing p_value
        p_val = 1 - chi2.cdf(LR, df = 1)

        # result dict
        result = {
            'observations': T,
            'breaches': N,
            'expected_breaches': p * T,
            'breach_ratio': p_hat,
            'LR': LR,
            'p_value': p_val,
            'passed_95%': p_val >= 0.05
        }

        return result
    
    
    # basel exception classifier
    def basel_exception_classifier(
            self,
            pnl,
            var
    ):
        """
        Basel exception classifier on the total number of observed breaches

        Emprical Basel thresholds are scaled from 250-day framework
            - 0-4 breaches -> green zone
            - 5-9 breaches -> yellow zone
            - 10+ breaches -> red zone
        """
        # observed breach vector
        I = self.compute_exception(
            pnl = pnl,
            var = var
        )

        # date length
        T = len(I)
        
        # number of observed breaches
        N = I.sum()

        # scaling threshold from Basel framework
        scaler = T / 250
        green_max = int(np.floor(4 * scaler))
        yellow_max = int(np.floor(9 * scaler))

        # defining Basel exception classes
        zone = 'Green' if N <= green_max else 'Yellow' if N <= yellow_max else 'Red'

        result = {
            'breaches': N,
            'zone': zone,
            'green_threshold': green_max,
            'yellow_threshold': yellow_max
        }

        return result
    
    # summary report
    def backtest_report(
            self,
            pnl,
            var
    ):
        """ Returns VaR backtesting report combining Kupiec test and Basel exception classifier """
        kupiec_report = self.kupiec_test(
            pnl = pnl,
            var = var
        )

        classifier_report = self.basel_exception_classifier(
            pnl = pnl,
            var = var
        )

        summary_report = {
            'Observations': kupiec_report['observations'],
            'Breaches': kupiec_report['breaches'],
            'Expected Breaches': kupiec_report['expected_breaches'],
            'Breach Ratio': kupiec_report['breach_ratio'],
            'Kupiec LR': kupiec_report['LR'],
            'Kupiec p-value': kupiec_report['p_value'],
            'Kupiec Pass (95%)': kupiec_report['passed_95%'],
            'Basel Zone': classifier_report['zone'],
            'Green Zone Threshold': classifier_report['green_threshold'],
            'Yellow Zone Threshold': classifier_report['yellow_threshold']
        }

        return pd.DataFrame.from_dict(
            summary_report, 
            orient = 'index',
            columns = ['Value']
        )
