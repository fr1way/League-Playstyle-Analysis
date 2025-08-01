import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def preprocess_lol_data(filepath_or_buffer):
    df = pd.read_csv(filepath_or_buffer)

    if df['start_utc'].dtype == object:
        df['start_utc'] = pd.to_datetime(df['start_utc'], utc=True)

    df = df.drop_duplicates(subset=['game_id', 'participant_id'], keep='first')
    core_stats = ['kills', 'deaths', 'assists', 'gold_per_min', 'vision_score']
    df = df.dropna(subset=[c for c in core_stats if c in df.columns])

    if 'queue' in df.columns:
        df = df[df['queue'] == 'Ranked Solo/Duo']

    df['vision_score'] = df['vision_score'].clip(lower=0, upper=200)

    feat_df = df.copy()
    feat_df['duration_mins'] = feat_df['duration'] / 60
    feat_df['kda_ratio'] = (feat_df['kills'] + feat_df['assists']) / (feat_df['deaths'] + 1)
    feat_df['damage_champ_per_min'] = feat_df['damage_to_champ'] / (feat_df['duration_mins'] + 1e-6)
    feat_df['vision_score_per_min'] = feat_df['vision_score'] / (feat_df['duration_mins'] + 1e-6)
    deaths_pm = feat_df['deaths'] / (feat_df['duration_mins'] + 1e-6)
    feat_df['survivability_score'] = 1 / (1 + deaths_pm)
    team_objs = (
        feat_df['team_towerKills'] + feat_df['team_dragonKills'] +
        feat_df['team_baronKills'] + feat_df['team_riftHeraldKills'] +
        feat_df['team_inhibitorKills']
    )
    feat_df['objective_score'] = feat_df['kill_participation'] * team_objs

    feat_df['fighting_score']  = feat_df[['kda_ratio','kill_participation','damage_champ_per_min']].mean(axis=1)
    feat_df['farming_score']   = feat_df['gold_per_min']
    feat_df['vision_score_pm'] = feat_df['vision_score_per_min']
    feat_df['objective_score'] = feat_df['objective_score']
    feat_df['survivability_score'] = feat_df['survivability_score']

    features = ['fighting_score','farming_score','vision_score_pm','objective_score','survivability_score']
    X = feat_df[features].fillna(0).replace([np.inf,-np.inf],0)

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=features)

    return X_scaled, df, scaler
