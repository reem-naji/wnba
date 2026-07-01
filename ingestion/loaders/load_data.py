import pandas as pd
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from ingestion.extractors.bbref_scraper import scrape_advanced_data, scrape_basic_data

SEASON = 2026

basic_stats = scrape_basic_data(SEASON)
basic_stats= basic_stats.loc[:, ~basic_stats.columns.duplicated()]

advanced_stats = scrape_advanced_data(SEASON)
advanced_stats= advanced_stats.loc[:, ~advanced_stats.columns.duplicated()]
advanced_stats.drop(advanced_stats.columns[-5], axis=1, inplace=True) # remove ' ' column , use indices to avoid keyerror
advanced_stats.drop(columns=['Team', 'Pos', 'G', 'MP'], inplace=True) # remove 'Team', 'Pos', 'G', 'MP', 'G', 'MP' columns

df = basic_stats.merge(advanced_stats, on=['Season','Player'])

decimal_cols = [col for col in df.columns if col not in ['Player', 'Team', 'Pos']]
for col in decimal_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df.to_csv(PROJECT_ROOT / 'ml' / 'data'/ f'player_data_{SEASON}.csv')
