import pandas as pd
import numpy as np

class PerformanceMetrics:
    """ Computes hedge effectiveness metrics using historical PnL time-series """
    ### Risk Metrics
    @staticmethod
    def volatility(pnl):
        """ Daily PnL volatility """
        return float(np.std(pnl, ddof = 1))
    
    @staticmethod
    def historical_var(pnl, confidence = 0.99):
        """ Historical VaR """
        q = -np.quantile(pnl, 1 - confidence)
        return float(q)
    
    @staticmethod
    def expected_shortfall(pnl, confidence = 0.99):
        """ Historical Expected Shortfall (CVaR) """
        var_threshold = np.quantile(pnl, 1 - confidence)
        tail_losses = pnl[pnl < var_threshold]

        return float(-np.mean(tail_losses))
    
    @staticmethod
    def max_drawdown(pnl):
        """ Maximum drawdown from cumulative PnL """
        cumulative = np.cumsum(pnl)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = cumulative - running_max

        return float(np.min(drawdown))
    
    ### Hedge Effectiveness Metrics
    @staticmethod
    def variance_reduction(pnl_pre, pnl_post):
        """ Percentage variance reduction after hedging in decimal points """
        variance_pre = np.var(pnl_pre, ddof = 1)
        variance_post = np.var(pnl_post, ddof = 1)

        return float(1 - variance_post / variance_pre)
    
    @staticmethod
    def var_reduction(pnl_pre, pnl_post, confidence = 0.99):
        """ Percentage VaR reduction after hedge in decimal points """
        var_pre = PerformanceMetrics.historical_var(
            pnl = pnl_pre,
            confidence = confidence
        )
        var_post = PerformanceMetrics.historical_var(
            pnl = pnl_post,
            confidence = confidence
        )

        return float(1 - var_post / var_pre)
    
    @staticmethod
    def cvar_reduction(pnl_pre, pnl_post, confidence = 0.99):
        """ Percentage CVaR reduction after hedge in decimal points """
        cvar_pre = PerformanceMetrics.expected_shortfall(
            pnl = pnl_pre,
            confidence = confidence
        )
        cvar_post = PerformanceMetrics.expected_shortfall(
            pnl = pnl_post,
            confidence = confidence
        )

        return float(1 - cvar_post / cvar_pre)
    
    ### Summary Report
    @staticmethod
    def hedge_effectiveness_report(
        pnl_pre,
        pnl_post,
        confidence = 0.99
    ):
        """ Summary report comparing pre- vs post-hedge scenarios under hedge effectiveness metrics """
        summary_report = {
            'Metric': [
                'Volatility',
                'Historical VaR',
                'Expected Shortfall',
                'Max Drawdown',
                'Variance Reduction %',
                'VaR Reduction %',
                'CVaR Reduction %'
            ],
            'Pre_Hedge': [
                PerformanceMetrics.volatility(pnl = pnl_pre),
                PerformanceMetrics.historical_var(pnl = pnl_pre, confidence = confidence),
                PerformanceMetrics.expected_shortfall(pnl = pnl_pre, confidence = confidence),
                PerformanceMetrics.max_drawdown(pnl = pnl_pre),
                np.nan,
                np.nan,
                np.nan
            ],
            'Post_Hedge': [
                PerformanceMetrics.volatility(pnl = pnl_post),
                PerformanceMetrics.historical_var(pnl = pnl_post, confidence = confidence),
                PerformanceMetrics.expected_shortfall(pnl = pnl_post, confidence = confidence),
                PerformanceMetrics.max_drawdown(pnl = pnl_post),
                np.nan,
                np.nan,
                np.nan
            ],
            'Improvement': [
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                PerformanceMetrics.variance_reduction(pnl_pre = pnl_pre, pnl_post = pnl_post),
                PerformanceMetrics.var_reduction(pnl_pre = pnl_pre, pnl_post = pnl_post, confidence = confidence),
                PerformanceMetrics.cvar_reduction(pnl_pre = pnl_pre, pnl_post = pnl_post, confidence = confidence)
            ]
        }

        return pd.DataFrame(summary_report)
    