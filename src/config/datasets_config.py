# risk-free proxy rates
CURVE_CONFIG = {
    # risk-free proxies
    'treasury': {
        '1M': 'DGS1MO',
        '3M': 'DGS3MO',
        '6M': 'DGS6MO',
        '1Y': 'DGS1',
        '2Y': 'DGS2',
        '5Y': 'DGS5',
        '10Y': 'DGS10',
        '30Y': 'DGS30'
    },
    # derivatives discounting curve
    'sofr': {
        'ON': 'SOFR'
        #'FEDFUNDS': 'FEDFUNDS'
    },
    # future proxies
    'futures': {
        'TBill3M': 'DTB3',
        'TBill6M': 'DTB6'
    }
}