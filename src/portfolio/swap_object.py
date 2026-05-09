# Container class for an IRSwap object
class IRSwap:
    """
    Vanilla fixed-for-floating interest rate swap

    Parameters:
        maturity: float
            Swap maturity in years (e.g. 10.0)
        fixed_rate: float
            Contractual fixed rate (e.g. 0.045 = 4.5%)
        notional: float
            Trade notional
        pay_receive: str
            'payer' -> pay fixed / receive float
            'receiver' -> receive fixed / pay float
        freq: int
            Fixed leg payment frequency (default = 2 -> semi-annual)           
    """
    def __init__(
            self,
            maturity: float,
            fixed_rate: float,
            notional: float,
            pay_receive: str,
            freq: int = 2
    ):
        # attributes
        self.maturity = maturity
        self.fixed_rate = fixed_rate
        self.notional = notional
        self.pay_receive = pay_receive
        self.freq = freq

        # validation
        if self.pay_receive not in ['payer', 'receiver']:
            raise ValueError('pay_receive must be either "payer" or "receiver"')
        
        if self.maturity <= 0:
            raise ValueError('maturity must be strictly positive!')
        
        #if self.notional <= 0:
        #    raise ValueError('notional must be strictly positive!')
        
        if self.freq <= 0:
            raise ValueError('frequency must be a positive integer number!')
    
    def direction_sign(self) -> int:
        """ Returns -1 for a payer swap, and +1 for a receiver swap """
        return -1 if self.pay_receive == 'payer' else 1
    
    def summary(self) -> str:
        """ Trade description """
        return (
            f"{self.pay_receive.upper()} IRS | "
            f"T={self.maturity}Y | "
            f"K={self.fixed_rate:.4%} | "
            f"N={self.notional:,.0f}"
        )
    