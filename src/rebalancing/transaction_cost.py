import numpy as np
import pandas as pd

class TransactionCostModel:
    """ 
    Incorporating the cost of trading hedge instruments into rolling-hedge decisions
    when conducting rolling hedges over simulation horizon 

    Cost Components:
        - Spread -> bid/ask price of the trade
        - Slippage -> trade size impacting the market price
        - Broker fee -> fixed cost per trade
    """
    def __init__(
            self,
            hedge_universe,
            custom_spread_bps: dict,
            impact_coeff: float = 0.10,
            daily_vol_bps: float = 7.0,
            fixed_cost: float = 50.00
    ):
        """
        Parameters:
            spread_bps: Bid-ask spreads in bps per hedge instrument
            impact_coeff: market impact coefficient
            daily_vol_bps: typical daily yield volatility
            fixed_cost: Fixed broker fee per trade (USD)
        """
        self.hedge_universe = hedge_universe
        self.custom_spread_bps = custom_spread_bps
        self.impact_coeff = impact_coeff
        self.daily_vol = daily_vol_bps / 10000
        self.fixed_cost = fixed_cost

        self.spread_bps = self._build_spread_dictionary(
            self.hedge_universe,
            self.custom_spread_bps
        )

    # spread dictionary builder
    def _build_spread_dictionary(self, hedge_universe, custom_spread_bps):
        """ Assigning bid/ask spreads defined in custom spread dictionary to spread dictionary with instrument names """
        spread_dict = {}
        
        for inst in hedge_universe.instruments:

            spread_dict[inst.name] = custom_spread_bps[inst.name.split('_')[-1]]

        return spread_dict

    # cost components
    def spread_cost(
            self,
            trade_dv01: float,
            instrument: str
    ) -> float:
        """ 
        Formula:
            Cost_spread = (spread * |Q|) / 2

            where:
                spread: bid/ask spread in decimal points per hedge instrument
                Q: trade size (DV01 notional) in dollars          
        """
        # converting spread in bps into decimal points
        spread = self.spread_bps[instrument] / 10000

        # compute spread cost
        cost_spr = (spread * abs(trade_dv01)) / 2

        return cost_spr
    
    def slippage_cost(
            self,
            trade_dv01: float
    ) -> float:
        """
        Formula:
            Cost_impact = H * sigma * (|Q|)^(1/2)

            where:
                H: liquidity coefficient (impact coefficient)
                sigma: daily yield volatility in deciaml points
                Q: trade size (DV01 notional) in USD
        """
        # compute slippage cost
        cost_slip = self.impact_coeff * self.daily_vol * np.sqrt(np.abs(trade_dv01))

        return cost_slip
    
    # trade-level cost
    def total_trade_cost(
            self,
            trade_dv01: float,
            instrument: str
    ) -> float:
        """
        Computing total cost per trade

        Cost_total = Cost_spread + Cost_slippage + Cost_fixed
        """
        cost_total = (
            self.spread_cost(trade_dv01 = trade_dv01, instrument = instrument) +
            self.slippage_cost(trade_dv01 = trade_dv01) +
            self.fixed_cost
        )
        return cost_total
    
    # portfolio rebalance cost
    def rebalance_cost(
            self,
            w_old: pd.Series,
            w_new: pd.Series,
            portfolio_dv01: float
    ) -> float:
        """
        Computing total cost of rebalancing hedge portfolio

        Formula:
            Q = Δw x DV01_portfolio

            where:
                Δw: delta hedge weights -> Δw = w_{t} - w_{t-1}
                DV01_portfolio: portfolio-level DV01 in USD        
        """
        # compute Δw
        delta_w = w_new - w_old

        rebalance_cost = 0.0

        for inst in delta_w.index:

            trade_dv01 = delta_w[inst] * portfolio_dv01
            rebalance_cost += self.total_trade_cost(
                trade_dv01 = trade_dv01,
                instrument = inst
            )

        return rebalance_cost
