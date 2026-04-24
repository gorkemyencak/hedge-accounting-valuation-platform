import pandas as pd

def build_term_sofr_curve(
        curves: dict
):
    """ 
    Building a synthetic short-end risk-free SOFR term structure using futures slope 
    
    Instruments used:
    ON -> SOFR
    1M -> interpolated
    3M -> 3M T-Bill proxy
    6M -> 6M T-Bill proxy    
    """

    sofr_on = curves['sofr']
    futures = curves['futures']

    t_3m = futures['TBill3M']
    t_6M = futures['TBill6M']

    df = pd.DataFrame(
        index = sofr_on.index
    )

    # Overnight anchor
    df['ON'] = sofr_on
    
    # 3M and &M from T-Bills
    df['3M'] = t_3m
    df['6M'] = t_6M

    # interpolate 1M between ON and 3M
    df['1M'] = df['ON'] + (df['3M'] - df['ON']) * (1/3)

    return df.round(2)

