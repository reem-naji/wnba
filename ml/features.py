import pandas as pd

STATS = ['WS', 'PER', 'USG%','TS%', 'pts_per_game', 'ast_per_game',
               'trb_per_game', 'blk_per_game', 'stl_per_game']

FEATURES = [f'{stat}_z' for stat in STATS]
TARGET = 'award_share'

def build_features(raw_df:pd.DataFrame, objective="train") -> pd.DataFrame:

    if objective == "train":
        df = raw_df[['Season', 'Player', 'Team', 'Pos', 'WS', 'PER', 'USG%','TS%', 'award_share']].copy()
    elif objective == "predict":
        df = raw_df[['Season', 'Player', 'Team', 'Pos', 'WS', 'PER', 'USG%', 'TS%']].copy()
    
    df['pts_per_game'] = raw_df['PTS'] / raw_df['G']
    df['stl_per_game'] = raw_df['STL'] / raw_df['G']
    df['ast_per_game'] = raw_df['AST'] / raw_df['G']
    df['trb_per_game'] = raw_df['TRB'] / raw_df['G']
    df['blk_per_game'] = raw_df['BLK'] / raw_df['G']

    for col in STATS:
        df[f'{col}_z'] = df.groupby('Season')[col].transform(
            lambda x: (x - x.mean()) / x.std()
        )

    return df