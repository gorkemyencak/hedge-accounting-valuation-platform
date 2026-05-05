import pandas as pd
import matplotlib.pyplot as plt

def plot_cumulative_pnl(
        pnl_explained: pd.DataFrame,
        pnl_actual: pd.DataFrame
) -> None:
    """ Plotting Actual vs Explained PnL """
    # joining explained and actuald dataframes
    df_pnl = pnl_actual.join(
        pnl_explained,
        how = 'inner'
    )

    # cumulative sum over entire history
    cumulative = df_pnl.cumsum()

    plt.figure(figsize = (12, 7))

    plt.plot(cumulative['Explained_PnL'], label = 'Explained_PnL')
    plt.plot(cumulative['Actual_PnL'], label = 'Actual_PnL')
    plt.title('Cumulative PnL - Actual vs Explained PnL', fontweight = 'bold')
    plt.xlabel('Date')
    plt.ylabel('PnL')
    plt.legend(loc = 'best')
    plt.show()


def plot_residuals(
        pnl_residual: pd.DataFrame
) -> None:
    """ Plotting Residual PnL """
    plt.figure(figsize = (12, 5))

    plt.plot(pnl_residual['Residual_PnL'])
    plt.title('Residual PnL (Unexplained PnL)', fontweight = 'bold')
    plt.xlabel('Date')
    plt.ylabel('Residual PnL')
    plt.show()


def plot_rolling_correlation(
        pnl_explained: pd.DataFrame,
        pnl_actual: pd.DataFrame,
        window: int = 10
) -> None:
    """ 
    Rolling correlation between Actual vs Explained PnL 
    
    > 0.9 -> Strong risk model
    0.7 < & < 0.9 -> Sufficient risk model
    < 0.7 -> Weak risk model
    """
    # rolling correlation
    rolling_corr = (
        pnl_explained['Explained_PnL']
        .rolling(window = window)
        .corr(pnl_actual['Actual_PnL'])
    )

    plt.figure(figsize=(12, 6))

    plt.plot(rolling_corr)
    plt.title(f'{window}-Day Rolling Correlation - Actual vs Explained', fontweight = 'bold')
    plt.xlabel('Date')
    plt.ylabel('Correlation')
    plt.show()
