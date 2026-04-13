from src.data.fred_downloader import FredCurveDownloader

class MarketLoader:

    def __init__(self):
        
        self.curves = {}
    

    def load_curve(
            self,
            curve_name
    ):
        
        loader = FredCurveDownloader(curve_name = curve_name)
        df = loader.download()
        self.curves[curve_name] = df

        return df