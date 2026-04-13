import os
import pandas as pd
from pathlib import Path
from fredapi import Fred
from src.config.datasets_config import CURVE_CONFIG

FRED_API_KEY = "41504b53ebf306bcd89ceb69bbd6eba8"

class FredCurveDownloader:
    """ Downloading US Treasury yield curve from FRED """
    def __init__(
            self,
            curve_name,
            data_dir = 'data/curves'
    ):
        
        if curve_name not in CURVE_CONFIG:
            raise ValueError(f"Unknown curve: {curve_name}")
        
        self.curve_name = curve_name
        
        project_root = Path(__file__).resolve().parents[2]
        self.data_dir = project_root / data_dir
        self.data_dir.mkdir(parents = True, exist_ok = True)

        self.file_path = self.data_dir / f"{curve_name}.csv"

        self.fred = Fred(api_key = FRED_API_KEY)

        self.series_map = CURVE_CONFIG[self.curve_name]

    
    def download(self) -> pd.DataFrame:

        if not self.file_path.exists():
            print(f"Downloading {self.curve_name} curve from FRED..")
            df = pd.DataFrame()

            for tenor, series in self.series_map.items():
                try:
                    df[tenor] = self.fred.get_series(series)
                except Exception as e:
                    print(f"{series} not available -> {e}")
                    df[tenor] = None

            df.index.name = 'Date'
            df = df.sort_index()

            # save the dataframe locally
            df.to_csv(self.file_path)

            return df  
        
        else:
            print(f"{self.curve_name} curve dataset already downloaded..")
            return pd.read_csv(self.file_path, index_col = 0, parse_dates = True)