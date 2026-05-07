import pandas as pd
from dataclasses import dataclass
from typing import List

from src.portfolio.swap_object import IRSwap

@dataclass
class HedgeInstrument:
    """ Container class describing a standard hedging IR swap """
    name: str
    maturity: float
    fixed_rate: float = 0.0
    freq: int = 2
    notional: float = 1_000_000
    pay_receive: str = 'receive'


class HedgeUniverse:
    """ A class for building hedge instrument universe, and providing liquid IR swaps spanning the yield curve """
    def __init__(
            self,
            instruments: List[HedgeInstrument]
    ):
        # attributes
        self.instruments = instruments  
        self.swaps = List[IRSwap] | None = None     # type: ignore


    # HedgeInstruments -> IRSwap objects
    def build_IRswaps(self):
        """ Converting hedge instruments into IR swap objects compatible with pricing/risk engine """
        swaps = []

        for hedgeinst in self.instruments:
            swap = IRSwap(
                maturity = hedgeinst.maturity,
                fixed_rate = hedgeinst.fixed_rate,
                notional = hedgeinst.notional,
                pay_receive = hedgeinst.pay_receive,
                freq = hedgeinst.freq
            )

            swaps.append(swap)
        
        self.swaps = swaps
        
        return swaps
    
    # HedgeUniverse summary table
    def summary(self):
        """ Returning the hedge universe IR swap instruments as a summary table """
        rows = []

        for hedgeinst in self.instruments:
            rows.append({
                'Instrument': hedgeinst.name,
                'Type': hedgeinst.pay_receive,
                'Maturity': hedgeinst.maturity,
                'Notional': hedgeinst.notional,
                'CouponFreq': hedgeinst.freq
            })
        
        return pd.DataFrame(rows)
