import pandas as pd
import numpy as np

from src.portfolio.swap_portfolio import SwapPortfolio

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
        Converting daily index into rebalance dates aligned to available market dates.
        We allow rebalance on the last available trading day of each period.

        T_rebalance = {t_0, t_n, t_2n, ..}

        where:
            n = rebalance window
        """
        # create series of ones with market index
        s = pd.Series(
            1,
            index = dates
        )

        # rebalance schedule
        rebalance_dates = (
            s.resample(rule = self.r_freq)
            .last()
            .dropna()
            .index
        )

        # guarantee they exist in curve index
        rebalance_dates = dates.intersection(rebalance_dates)

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
        """ 
        Build daily DV01 of the hedge portfolio (parallel DV01 per instrument) 
        Formula:
            portfolio DV01 = sum_i (weight_i * DV01_i) -> (dates x tenors)
        """
        # get dates
        dates = df_curve.index

        # hedge dv01 container
        hedge_portfolio_dv01 = pd.DataFrame(
            index = dates,
            columns = key_rate_tenors,
            dtype = float
        )

        # to compute DV01 per swap individually
        dv01_list = []

        for inst in hedge_universe.instruments:
            
            # create single swap portfolio for each hedge instrument 
            single_swap_portfolio = SwapPortfolio(swaps = [inst])

            # key-rate DV01 timeseries for each hedge instrument
            dv01_ts = single_swap_portfolio.portfolio_dv01(
                df_curve = df_curve,
                shock_type = 'key_rate',
                key_rate_tenors = key_rate_tenors
            )

            # create column MultiIndex: (instrument, tenor)
            dv01_ts.columns = pd.MultiIndex.from_product(
                [[inst.name], dv01_ts.columns]
            )

            dv01_list.append(dv01_ts)
        
        # concatenate all instruments along columns -> (dates x (instruments x tenors))
        hedge_dv01_matrix = pd.concat(dv01_list, axis = 1)
        
        # apply hedge weights between rebalances
        rebalance_dates = hedge_positions.index

        for i, reb_date in enumerate(rebalance_dates):

            #start = reb_date
            start_idx = dates.get_loc(reb_date) + 1

            if start_idx >= len(dates):
                continue

            start = dates[start_idx]

            end = dates[-1] if i == len(rebalance_dates) - 1 else rebalance_dates[i+1]

            weights = hedge_positions.loc[reb_date]

            mask = (dates >= start) & (dates < end)

            inst_dv01 = hedge_dv01_matrix.loc[mask]

            weighted = sum(
                weights[name] * inst_dv01[name]
                for name in weights.index
            )

            hedge_portfolio_dv01.loc[mask] = weighted
        
        return hedge_portfolio_dv01

    # daily portfolio DV01 timeseries
    def _build_daily_portfolio_dv01(self, portfolio, df_curve, key_rate_tenors):
        dates = df_curve.index
        rebalance_dates = self.hedge_positions.index # type:ignore

        # dv01 at rebalance dates
        portfolio_dv01_reb = portfolio.portfolio_dv01(
            df_curve = df_curve.loc[rebalance_dates],
            shock_type = 'key_rate',
            key_rate_tenors = key_rate_tenors
        )

        # forward fill between rebalances
        portfolio_dv01_daily = portfolio_dv01_reb.reindex(dates).ffill()
        return portfolio_dv01_daily

    
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
            PnL_portfolio = -sum(DV01_{t-1} x dy_{t} * 10000)

        Hedge Assets PnL Formula:
            PnL_hedge = +sum(DV01_{t-1} x dy_{t} * 10000)

        Total PnL Formula:
            PnL_total = PnL_portfolio + PnL_hedge - transaction_costs
        """
        # delta yield changes
        dy = self._compute_yield_changes(yield_curve = df_curve) * 10000
        
        # portfolio DV01
        portfolio_dv01 = self._build_daily_portfolio_dv01(
            portfolio = portfolio,
            df_curve = df_curve,
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
                df_curve = df_curve.loc[:date],
                rebalance_date = date
            )

            # converting new_hedge numpy array into series
            new_hedge = pd.Series(
                new_hedge,
                index = hedge_universe.instrument_names
            )

            # turnover -> compute trades vs previous hedge
            trade = new_hedge if current_hedge is None else new_hedge - current_hedge

            # proxy portfolio DV01 magnitude for cost scaling
            portfolio_dv01_today = portfolio.portfolio_dv01(
                df_curve = df_curve.loc[:date],
                shock_type = 'parallel'
            ).iloc[-1].values[0]

            # transaction cost
            if current_hedge is None:
                cost = self.tc_model.rebalance_cost(
                    w_old = pd.Series(0, index = new_hedge.index),
                    w_new = new_hedge,
                    portfolio_dv01 = portfolio_dv01_today
                )
            else:
                cost = self.tc_model.rebalance_cost(
                    w_old = current_hedge,
                    w_new = new_hedge,
                    portfolio_dv01 = portfolio_dv01_today
                )

            # setting storage attributes
            hedge_positions[date] = new_hedge
            trades[date] = trade
            costs[date] = cost

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
    