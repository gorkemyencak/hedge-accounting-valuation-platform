import pandas as pd
import matplotlib.pyplot as plt


def plot_discount_curve(
        df_curve: pd.DataFrame,
        date
):
    """
    Plotting discount curve for a single date

    This plot should verify that:
        - discount curve should start near 1
        - be monotonically decreasing
        - never takes negative values 
    """
    # ensuring that date parameter exists in the discount curve index
    date = pd.to_datetime(date)
    if date not in df_curve.index:
        raise ValueError(f'Date: {date} not found in the df_curve index!')
    
    # extracting curve
    curve = df_curve.loc[date]
    maturities = curve.index.astype(float)

    # plotting the discount curve
    plt.figure(figsize = (10, 5))
    plt.plot(
        maturities,
        curve.values,
        marker = 'o',
        markersize = 3
    )
    plt.title(f'Discount Curve as of {date.date()}', fontweight = 'bold')
    plt.xlabel('Maturity (in years)')
    plt.ylabel('Discount Factor')
    plt.show()


def plot_zero_curve(
        zero_curve: pd.DataFrame,
        date
):
    """
    Plotting zero curve for a single date
    
    This plot should verify that:
        - no spikes across different maturities
        - it should range between expected values
    """
    # ensuring that date parameter exists in the zero curve index
    date = pd.to_datetime(date)
    if date not in zero_curve.index:
        raise ValueError(f'Date: {date} not found in the zero_curve index!')
    
    # extracting curve
    curve = zero_curve.loc[date]
    maturities = curve.index.astype(float)

    # plotting the zero curve
    plt.figure(figsize = (10, 5))
    plt.plot(
        maturities,
        curve.values,
        marker = 'o',
        markersize = 3
    )
    plt.title(f'Zero Curve as of {date.date()}', fontweight = 'bold')
    plt.xlabel('Maturity (in years)')
    plt.ylabel('Rate')
    plt.show()


def plot_forward_curve(
        forward_curve: pd.DataFrame,
        date
):
    """
    Plotting forward curve for a single date
    """
    # ensuring that date parameter exists in the forward curve index
    date = pd.to_datetime(date)
    if date not in forward_curve.index:
        raise ValueError(f'Date: {date} not found in the forward_curve index!')
    
    # extracting curve
    curve = forward_curve.loc[date]
    maturities = curve.index.astype(float)

    # plotting the forward curve
    plt.figure(figsize = (10, 5))
    plt.plot(
        maturities,
        curve.values,
        marker = 'o',
        markersize = 3
    )
    plt.title(f'Forward Curve as of {date.date()}', fontweight = 'bold')
    plt.xlabel('Forward Start (in years)')
    plt.ylabel('Rate')
    plt.show()
    

def plot_curve_evolution(
        curve,
        curve_name,
        max_dates = 5
):
    """
    Plotting how a curve evolves across multiple dates
    """
    # most recent N dates
    recent_dates = curve.index[-max_dates:]
    
    plt.figure(figsize = (10, 5))

    for date in recent_dates:
        plt.plot(
            curve.columns,
            curve.loc[date],
            label = str(date.date()),
            alpha = 0.7
        )
    
    plt.title(f'{curve_name} Curve Evolution', fontweight = 'bold')
    plt.xlabel('Maturity (in years)')
    plt.ylabel('Rate')
    plt.legend(loc = 'best')
    plt.show()
