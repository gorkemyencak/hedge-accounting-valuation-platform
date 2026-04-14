import numpy as np

# coupon payments
def get_coupon_times(
        maturity,
        freq = 2
):
    """ computing the number of coupon payments will be made throughout the bond maturity """
    n_payments = maturity * freq
    times_arr = np.array([(i+1) / freq for i in range(n_payments)])

    return times_arr

# bond valuation
def bond_pricing_from_df(
        coupon_rate,
        maturity,
        dfs,
        freq = 2
):
    """
    Pricing a bullet bond using discount factors

    Parameters
    coupon_rate: float (in percent)
    dfs: dict {time_in_years: discount_factor}
    """
    coupon = 100 * (coupon_rate / 100 / freq)
    times = get_coupon_times(
        maturity = maturity,
        freq = freq
    )

    price = 0.0

    for t in times[:-1]:
        price += coupon * dfs[t]
    
    # final payment for a bullet bond -> coupon + principal
    price += (100 + coupon) * dfs[times[-1]]

    return price
