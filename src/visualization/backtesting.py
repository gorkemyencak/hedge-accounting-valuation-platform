import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class BacktestVisualizer:
    """ Visualization module for hedge backtesting results """
    ### Backtesting Plots
    @staticmethod
    def plot_cumulative_pnl(
        pnl_pre,
        pnl_post
    ):
        """ Plot cumulative PnL pre- vs post-hedging """
        cumulative_pre = pnl_pre.cumsum()
        cumulative_post = pnl_post.cumsum()

        plt.figure(figsize = (12, 5))
        plt.plot(
            cumulative_pre,
            label = 'Pre-Hedge'
        )
        plt.plot(
            cumulative_post,
            label = 'Post-Hedge'
        )
        plt.title('Cumulative PnL', fontweight = 'bold')
        plt.legend(loc = 'best')
        #plt.ylim(None)
        plt.show()

    
    @staticmethod
    def plot_rolling_vol(
        pnl_pre,
        pnl_post,
        window = 10
    ):
        """ Plot rolling volatility comparison (pre- vs post-hedging) """
        volatility_pre = pnl_pre.rolling(window = window).std()
        volatility_post = pnl_post.rolling(window =window).std()

        plt.figure(figsize = (12, 5))
        plt.plot(
            volatility_pre,
            label = 'Pre-Hedge'
        )
        plt.plot(
            volatility_post,
            label = 'Post-Hedge'
        )
        plt.title(f'PnL Rolling Volatility ({window}d)', fontweight = 'bold')
        plt.legend(loc = 'best')
        plt.show()

    
    @staticmethod
    def plot_drawdown(
        pnl_pre,
        pnl_post
    ):
        """ Plot drawdown from cumulative PnL to compare pre- vs post-hedging """
        def compute_drawdown(pnl):

            cum = pnl.cumsum()
            running_max = cum.cummax()

            return cum - running_max
        
        drawdown_pre = compute_drawdown(pnl = pnl_pre)
        drawdown_post = compute_drawdown(pnl = pnl_post)

        plt.figure(figsize = (12, 5))
        plt.plot(
            drawdown_pre,
            label = 'Pre-Hedge'
        )
        plt.plot(
            drawdown_post,
            label = 'Post-Hedge'
        )
        plt.title('Cumulative PnL Drawdown', fontweight = 'bold')
        plt.legend(loc = 'best')
        plt.show()

    
    @staticmethod
    def plot_var_overlay(
        pnl,
        var_series,
        confidence: float = 0.99
    ):
        """ 
        Plot daily PnL against VaR forecast 
        
        Breach condition:
            PnL_t < -VaR_t
        """
        plt.figure(figsize = (12, 5))
        plt.plot(
            pnl,
            label = 'Daily PnL'
        )
        plt.plot(
            -var_series,
            label = f'-VaR ({str(100 * confidence)}%)'
        )
        plt.title('PnL vs VaR', fontweight = 'bold')
        plt.legend(loc = 'best')
        plt.axhline(0, linestyle = '--', color = 'black')
        plt.show()

    
    @staticmethod
    def plot_var_breaches(
        pnl,
        var_series
    ):
        """ Plot VaR breach timeline """
        breaches = pnl < -var_series

        plt.figure(figsize = (12, 5))
        plt.plot(
            pnl,
            label = 'Daily PnL'
        )
        plt.scatter(
            pnl.index[breaches],
            pnl[breaches],
            marker = 'o',
            s = 20,
            color = 'red',
            edgecolors = 'red',
            label = 'Breach'
        )
        plt.title('VaR Breach Timeline', fontweight = 'bold')
        plt.legend(loc = 'best')
        plt.axhline(0, linestyle = '--', color = 'black')
        plt.show()


    @staticmethod
    def plot_cumulative_breaches(
        pnl,
        var_series
    ):
        """ Cumulative exception monitoring """
        breaches = (pnl < -var_series).astype(int)
        cum_breaches = breaches.cumsum()

        plt.figure(figsize = (12, 5))
        plt.plot(
            cum_breaches,
            label = 'Cumulative Breaches'
        )
        plt.title('Cumulative VaR Breaches', fontweight = 'bold')
        plt.legend(loc = 'best')
        plt.show()

    
    # Full Report Pipeline
    @staticmethod
    def backtest_visualizer_pipeline(
        pnl_pre,
        pnl_post,
        window_size,
    ):
        """ Generate full backtest visualizer plots """
        BacktestVisualizer.plot_cumulative_pnl(
            pnl_pre = pnl_pre,
            pnl_post = pnl_post
        )
        BacktestVisualizer.plot_rolling_vol(
            pnl_pre = pnl_pre,
            pnl_post = pnl_post,
            window = window_size
        )
        BacktestVisualizer.plot_drawdown(
            pnl_pre = pnl_pre,
            pnl_post = pnl_post
        )
    

    @staticmethod
    def var_backtest_visualizer_pipeline(
        pnl,
        var,
        confidence,
    ):
        """ Generate full backtest visualizer plots """
        BacktestVisualizer.plot_var_overlay(
            pnl = pnl,
            var_series = var,
            confidence = confidence
        )
        BacktestVisualizer.plot_var_breaches(
            pnl = pnl,
            var_series = var
        )
        BacktestVisualizer.plot_cumulative_breaches(
            pnl = pnl,
            var_series = var
        )
       
