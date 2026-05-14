import pandas as pd
import numpy as np

class RollingHedgeEngine:
    """
    Rolling Hedge Engine builds a time-evolving hedging simulator by re-estimating hedge regularly,
    rebalancing hedge positions, and tracking PnL of hedging strategy considering the transaction costs 
    associated with the selected hedge strategy.

    The module is designed to address:
        - return of hedged portfolio
        - rebalancing schedule
        - rolling hedge ratio estimation
        - turnover
        - cumulative hedged portfolio PnL    
    """
    def __init__(
            self,
            hedge_engine,
            transaction_cost_model,
            rebalance_frequency: str = 'ME'
    ):
        # attributes
        self.hedge_engine = hedge_engine
        self.tc_model = transaction_cost_model
        self.r_freq = rebalance_frequency

        # storage
        self.hedge_positions = None
        self.trade_log = None
        self.cost_series = None
        self.pnl_series = None

    # rebalance schedule
    def _generate_rebalance_schedule(
            self,
            dates
    ):
        """
        Converting daily index into rebalance dates

        T_rebalance = {t_0, t_n, t_2n, ..}

        where:
            n = rebalance window
        """
        # rebalance schedule
        rebalance_dates = (
            pd.Series(index = dates)
            .resample(rule = self.r_freq)
            .last()
            .index
        )

        return rebalance_dates
    
    # daily yield moves
    def _compute_yield_changes(
            self,
            yield_curve
    ):
        """ Δy_t used for daily PnL """
        return yield_curve.diff().dropna()
    
    # daily hedge DV01 timeseries
    def _build_daily_hedge_dv01(
            self,
            hedge_universe,
            df_curve,
            hedge_positions,
            key_rate_tenors            
    ):
        """ Converts piecewise-constant hedge notionals into daily DV01 timeseries """
        # hedge trade DV01 timeseries (per unit notional)
        trade_dv01_ts = hedge_universe.trade_dv01(
            df_curve = df_curve,
            shock_type = 'key_rate',
            key_rate_tenors = key_rate_tenors            
        )

        hedge_dv01_daily = pd.DataFrame(
            0,
            index = df_curve.index,
            columns = trade_dv01_ts.columns
        )

        for i, reb_date in enumerate(hedge_positions.index):

            start = reb_date
            end = hedge_positions.index[i+1] if i+1 < len(hedge_positions) else df_curve.index[-1]

            weights = hedge_positions.loc[reb_date].values

            mask = (
                (hedge_dv01_daily.index >= start) & (hedge_dv01_daily.index <= end)
            )

            hedge_dv01_daily.loc[mask] = trade_dv01_ts.loc[mask].values * weights
        
        return hedge_dv01_daily
    
    # daily PnL computation
    def compute_pnl(
            self,
            portfolio,
            hedge_universe,
            df_curve,
            key_rate_tenors
    ):
        """ 
        Computing hedged PnL time series 
        
        Portfolio-level PnL Formula:
            PnL_portfolio = -sum(DV01_{t-1} x dy_{t})

        Hedge Assets PnL Formula:
            PnL_hedge = +sum(DV01_{t-1} x dy_{t})

        Total PnL Formula:
            PnL_total = PnL_portfolio + PnL_hedge - transaction_costs
        """
        # delta yield changes
        dy = self._compute_yield_changes(yield_curve = df_curve)

        # portfolio DV01
        portfolio_dv01 = portfolio.portfolio_dv01(
            df_curve = df_curve,
            shock_type = 'key_rate',
            key_rate_tenors = key_rate_tenors
        )

        # hedge DV01
        hedge_dv01 = self._build_daily_hedge_dv01(
            hedge_universe = hedge_universe,
            df_curve = df_curve,
            hedge_positions = self.hedge_positions,
            key_rate_tenors = key_rate_tenors
        )

        # align indices
        portfolio_dv01 = portfolio_dv01.loc[dy.index]
        hedge_dv01 = hedge_dv01.loc[dy.index]

        # unhedged PnL (portfolio)
        pnl_portfolio = -(portfolio_dv01.shift(periods = 1) * dy).sum(axis = 1)

        # hedge PnL
        pnl_hedge = +(hedge_dv01.shift(periods = 1) * dy).sum(axis = 1)

        # transaction costs
        cost = self.cost_series.reindex(dy.index).fillna(0) # type: ignore

        # compute total PnL
        total_pnl = pnl_portfolio + pnl_hedge - cost

        self.pnl_series = total_pnl.cumsum()

        return self.pnl_series


    # rolling hedge simulator
    def rolling_hedge_simulator(
            self,
            portfolio,
            hedge_universe,
            df_curve,
            key_rate_tenors
    ):
        """ Dynamic rolling hedge simulator """
        # get rebalance schedule
        dates = df_curve.index
        rebalance_schedule = self._generate_rebalance_schedule(dates = dates)

        # storage attributes
        current_hedge = None
        hedge_positions = {} 
        trades = {}
        costs = {}

        for date in rebalance_schedule:

            # compute optimal hedge for a given rebalance date
            new_hedge, _ = self.hedge_engine.solve_opt_hedge(
                portfolio = portfolio,
                hedge_universe = hedge_universe,
                df_curve = df_curve,
                rebalance_date = date
            )

            # turnover -> compute trades vs previous hedge
            trade = new_hedge if current_hedge is None else new_hedge - current_hedge

            # proxy portfolio DV01 magnitude for cost scaling
            portfolio_dv01_today = portfolio.portfolio_dv01(
                df_curve = df_curve.loc[:date],
                shock_type = 'parallel'
            ).iloc[-1].values[0]

            # transaction cost
            cost = self.tc_model.rebalance_cost(
                w_old = current_hedge,
                w_new = new_hedge,
                portfolio_dv01 = portfolio_dv01_today
            )

            # setting storage attributes
            hedge_positions[date] = new_hedge
            trades[date] = trade
            costs[dates] = cost

            current_hedge = new_hedge
        
        self.hedge_positions = pd.DataFrame.from_dict(
            hedge_positions,
            orient = 'index'
        )
        self.trade_log = pd.DataFrame.from_dict(
            trades,
            orient = 'index'
        )
        self.cost_series = pd.Series(
            costs,
            name = 'TransactionCost'
        )

        return self
    


