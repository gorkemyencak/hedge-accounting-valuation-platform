import pandas as pd

from src.config.datasets_config import CURVE_CONFIG

from src.data.fred_downloader import FredCurveDownloader

class MarketLoader:

    def __init__(self):
        
        self.raw_curves = {}
        self.clean_curves = {}
    
    # download raw curves
    def download_curves(self):

        for curve in CURVE_CONFIG.keys():
            loader = FredCurveDownloader(
                curve_name = curve
            )

            self.raw_curves[curve] = loader.download()

    # clean single curve
    def _clean_curve(
            self,
            df
    ):
        
        df = df.copy()

        # convert to numeric columns
        df = df.apply(pd.to_numeric, errors = 'coerce')

        # handling missing values (forward fill)
        df = df.ffill()

        # dropping empty rows
        df = df.dropna(how = 'all')

        return df
    
    # clean all curves
    def clean_all_curves(self):

        for name, df_curve in self.raw_curves.items():
            self.clean_curves[name] = self._clean_curve(df = df_curve)

    # date aligment across all curves
    def align_dates(self):

        df_merged = (
            pd.concat(
                self.clean_curves.values(),
                axis = 1,
                keys = self.clean_curves.keys()
            )
            .ffill()
            .dropna()
        )

        return df_merged
        
    # market loader pipeline
    def loader_pipeline(self):
        
        self.download_curves()
        self.clean_all_curves()
        df_curves = self.align_dates()

        return df_curves