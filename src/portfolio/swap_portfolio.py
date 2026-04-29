# Container class of a swap portfolio consisting of a list of IR swaps
import pandas as pd

from src.pricing.swap_pricing_engine import swap_npv

from src.risk.swap_risk import swap_dv01_pipeline

class SwapPortfolio:
    """
    Container for a portfolio of interest rate swaps
    """
    def __init__(
            self,
            swaps: list
    ):
        
        self.swaps = swaps

    # pricing each swap inside the IR swap portfolio
    def price_trades(
            self,
            df_curve: pd.DataFrame
    ) -> pd.DataFrame:
        """ Returns NPV of each trade as time series """
        results = pd.DataFrame(index = df_curve.index)

        for i, swap in enumerate(self.swaps):

            results[f'Trade_{i}'] = swap_npv(
                df_curve = df_curve,
                maturity = swap.maturity,
                fixed_rate = swap.fixed_rate,
                freq = swap.freq,
                notional = swap.notional
            )
        
        return results
    
    # computing total portfolio NPV
    def portfolio_npv(
            self,
            df_curve: pd.DataFrame
    ) -> pd.DataFrame:
        """ Total portfolio NPV as time series """
        trade_npv = self.price_trades(
            df_curve = df_curve
        )
        
        total_npv = trade_npv.sum(axis = 1)

        total_npv = pd.DataFrame(
            total_npv,
            columns = ['Portfolio_NPV'] 
        )

        return total_npv
    
    # trade-level DV01
    def trade_dv01(
            self,
            df_curve: pd.DataFrame,
            shock_bps: float = 1.0,
            shock_type: str = 'parallel',
            key_rate_tenors: list[float] | None = None,
            multi_tenor_dict: dict[float, float] | None = None
    ):
        """ 
        Returns trade-level DV01 
        
        Output depends on the shock type:
            parallel -> DataFrame indexed by date
            key_rate -> DataFrame indexed by tenor
            multi_tenor -> Series (one value per trade)
        """
        dv01s = {}

        # parallel DV01
        if shock_type == 'parallel':

            results = pd.DataFrame(index = df_curve.index)

            for i, swap in enumerate(self.swaps):

                results[f'Trade_{i}'] = swap_dv01_pipeline(
                    df_curve = df_curve,
                    maturity = swap.maturity,
                    fixed_rate = swap.fixed_rate,
                    freq = swap.freq,
                    notional = swap.notional,
                    shock_bps = shock_bps,
                    shock_type = shock_type,
                    key_rate_tenors = key_rate_tenors,
                    multi_tenor_dict = multi_tenor_dict 
                )

            return results.round(1)

        # key_rate DV01
        elif shock_type == 'key_rate':

            if key_rate_tenors is None:
                raise ValueError(f"key_rate_tenors must be provided!")
            
            dv01s = {}

            for i, swap in enumerate(self.swaps):

                dv01s[f'Trade_{i}'] = swap_dv01_pipeline(
                    df_curve = df_curve,
                    maturity = swap.maturity,
                    fixed_rate = swap.fixed_rate,
                    freq = swap.freq,
                    notional = swap.notional,
                    shock_bps = shock_bps,
                    shock_type = shock_type,
                    key_rate_tenors = key_rate_tenors,
                    multi_tenor_dict = multi_tenor_dict 
                )
            
            return pd.concat(dv01s, axis = 1).round(1)
        
        # multi-tenor DV01
        elif shock_type == 'multi_tenor':

            if multi_tenor_dict is None:
                raise ValueError(f"multi_tenor_dict must be provided!")
            
            results = pd.DataFrame(index = df_curve.index)

            for i, swap in enumerate(self.swaps):

                results[f'Trade_{i}'] = swap_dv01_pipeline(
                    df_curve = df_curve,
                    maturity = swap.maturity,
                    fixed_rate = swap.fixed_rate,
                    freq = swap.freq,
                    notional = swap.notional,
                    shock_bps = shock_bps,
                    shock_type = shock_type,
                    key_rate_tenors = key_rate_tenors,
                    multi_tenor_dict = multi_tenor_dict 
                )
            
            return results.round(1)#pd.Series(dv01s).round(1)
        
        else:
            raise ValueError("Unknown shock_type. Please choose: 'parallel' | 'key_rate' | 'multi_tenor'")
        
    
    # portfolio-level DV01
    def portfolio_dv01(
            self,
            df_curve: pd.DataFrame,
            shock_bps: float = 1.0,
            shock_type: str = 'parallel',
            key_rate_tenors: list[float] | None = None,
            multi_tenor_dict: dict[float, float] | None = None
    ) -> pd.DataFrame:
        """ Returns portfolio-level DV01 """
        dv01s_by_trade = self.trade_dv01(
            df_curve = df_curve,
            shock_bps = shock_bps,
            shock_type = shock_type,
            key_rate_tenors = key_rate_tenors,
            multi_tenor_dict = multi_tenor_dict
        )

        dv01_portfolio = dv01s_by_trade.sum(axis = 1)
        dv01_portfolio = pd.DataFrame(
            dv01_portfolio,
            columns = ['Portfolio_DV01']
        )

        return dv01_portfolio


    def summary(self) -> pd.DataFrame:
        """ Portfolio description """
        rows = []

        for i, swap in enumerate(self.swaps):
            rows.append(
                {
                    'TradeID': i,
                    'Type': swap.pay_receive,
                    'Maturity': swap.maturity,
                    'FixedRate': swap.fixed_rate,
                    'Notional': swap.notional,
                    'CouponFrequency': swap.freq
                }
            )

        return pd.DataFrame(rows)
