from pathlib import Path
import pandas as pd
from skrub import TableReport

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = PROJECT_ROOT / "data/"
DEFAULT_SAMPLE = PROJECT_ROOT / "data/interim/final_sample.csv"


df = pd.read_csv(DEFAULT_SAMPLE)
report = TableReport(df)

# O para guardarlo en HTML:
report.write_html(DEFAULT_DIR / "EDA.html")
