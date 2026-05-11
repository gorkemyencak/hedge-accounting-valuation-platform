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

    
    # Full Report Pipeline
    @staticmethod
    def backtest_visualizer_pipeline(
        pnl_pre,
        pnl_post,
        window_size
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
