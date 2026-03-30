# train.py
import pandas as pd
import numpy as np
from xgboost import XGBRegressor, XGBClassifier
import joblib # .pkl files save karne ke liye
import os

print("⏳ Loading data and training models... Please wait.")

# 1. Load Data
df = pd.read_csv('data/IPL.csv')

# Safe Exact Column Mapping
df['runs_batter'] = pd.to_numeric(df.get('runs_batter', 0), errors='coerce').fillna(0)
df['runs_bowler'] = pd.to_numeric(df.get('runs_bowler', 0), errors='coerce').fillna(0)
df['runs_total'] = pd.to_numeric(df.get('runs_total', 0), errors='coerce').fillna(0)
df['ball'] = pd.to_numeric(df.get('ball', 0), errors='coerce').fillna(0)
df['over_num'] = pd.to_numeric(df.get('over', 0), errors='coerce').fillna(0).astype(int)

if 'player_out' in df.columns:
    df['is_wicket'] = df['player_out'].notna().astype(int)
else:
    df['is_wicket'] = 0

# 2. TARGET ENCODING 
inn1_scores = df[df['innings'] == 1].groupby(['match_id', 'venue', 'batting_team', 'bowling_team'])['runs_total'].sum().reset_index()

venue_strength = inn1_scores.groupby('venue')['runs_total'].mean().to_dict()
bat_team_strength = inn1_scores.groupby('batting_team')['runs_total'].mean().to_dict()
bowl_team_strength = inn1_scores.groupby('bowling_team')['runs_total'].mean().to_dict()
global_avg = inn1_scores['runs_total'].mean()

df['bat_team_id'] = df['batting_team'].map(bat_team_strength).fillna(global_avg)
df['bowl_team_id'] = df['bowling_team'].map(bowl_team_strength).fillna(global_avg)
df['venue_id'] = df['venue'].map(venue_strength).fillna(global_avg)

# --- ML SCORE MODEL ---
inn1 = df[df['innings'] == 1].copy()
inn1_overs = inn1.groupby(['match_id', 'bat_team_id', 'bowl_team_id', 'venue_id', 'over_num']).agg({'runs_total': 'sum', 'is_wicket': 'sum'}).reset_index()
inn1_overs['curr_score'] = inn1_overs.groupby('match_id')['runs_total'].cumsum()
inn1_overs['curr_wkts'] = inn1_overs.groupby('match_id')['is_wicket'].cumsum()

final_scores = inn1.groupby('match_id')['runs_total'].sum().reset_index().rename(columns={'runs_total': 'final_score'})
inn1_overs = inn1_overs.merge(final_scores, on='match_id')

inn1_overs['crr'] = np.where((inn1_overs['over_num'] + 1) > 0, inn1_overs['curr_score'] / (inn1_overs['over_num'] + 1), 0)
inn1_overs['wkt_resource'] = np.sqrt(np.maximum(0, 10 - inn1_overs['curr_wkts']) / 10.0)
inn1_overs['expected_rpo'] = (inn1_overs['venue_id'] + inn1_overs['bat_team_id']) / 40.0
inn1_overs['matchup_factor'] = np.where(inn1_overs['expected_rpo'] > 0, inn1_overs['crr'] / inn1_overs['expected_rpo'], 1.0)

X_score = inn1_overs[['bat_team_id', 'bowl_team_id', 'venue_id', 'curr_score', 'curr_wkts', 'over_num', 'crr', 'wkt_resource', 'matchup_factor']]
y_score = inn1_overs['final_score']
score_model = XGBRegressor(n_estimators=150, max_depth=5, learning_rate=0.05, random_state=42).fit(X_score, y_score)

# --- ML WIN MODEL ---
inn2 = df[df['innings'] == 2].copy()
inn2_overs = inn2.groupby(['match_id', 'batting_team', 'bat_team_id', 'bowl_team_id', 'venue_id', 'over_num']).agg({'runs_total': 'sum', 'is_wicket': 'sum'}).reset_index()
inn2_overs['curr_score'] = inn2_overs.groupby('match_id')['runs_total'].cumsum()
inn2_overs['curr_wkts'] = inn2_overs.groupby('match_id')['is_wicket'].cumsum()

targets = final_scores.rename(columns={'final_score': 'target'})
inn2_overs = inn2_overs.merge(targets, on='match_id', how='inner')

if 'match_won_by' in df.columns:
    match_winners = df[['match_id', 'match_won_by']].drop_duplicates()
    inn2_overs = inn2_overs.merge(match_winners, on='match_id', how='left')
    inn2_overs['is_win'] = (inn2_overs['batting_team'] == inn2_overs['match_won_by']).astype(int)
else:
    inn2_final = inn2.groupby('match_id')['runs_total'].sum().reset_index().rename(columns={'runs_total': 'inn2_score'})
    inn2_overs = inn2_overs.merge(inn2_final, on='match_id')
    inn2_overs['is_win'] = (inn2_overs['inn2_score'] > inn2_overs['target']).astype(int)

inn2_overs['runs_needed'] = inn2_overs['target'] - inn2_overs['curr_score']
inn2_overs['balls_left'] = 120 - ((inn2_overs['over_num'] + 1) * 6)
inn2_overs['rrr'] = np.where(inn2_overs['balls_left'] > 0, (inn2_overs['runs_needed'] / inn2_overs['balls_left']) * 6, 0.0)
inn2_overs['crr'] = np.where((inn2_overs['over_num'] + 1) > 0, inn2_overs['curr_score'] / (inn2_overs['over_num'] + 1), 0)
inn2_overs['pressure_gap'] = inn2_overs['rrr'] - inn2_overs['crr']
inn2_overs['wkt_resource'] = np.sqrt(np.maximum(0, 10 - inn2_overs['curr_wkts']) / 10.0)

X_win = inn2_overs[['bat_team_id', 'bowl_team_id', 'venue_id', 'target', 'curr_score', 'curr_wkts', 'over_num', 'runs_needed', 'balls_left', 'rrr', 'pressure_gap', 'wkt_resource']]
y_win = inn2_overs['is_win']
win_model = XGBClassifier(n_estimators=150, max_depth=5, learning_rate=0.05, random_state=42).fit(X_win, y_win)

# --- 3. SAVE EVERYTHING TO .PKL ---
os.makedirs('models', exist_ok=True)

# Save Encodings
encodings = {
    'venue_strength': venue_strength,
    'bat_team_strength': bat_team_strength,
    'bowl_team_strength': bowl_team_strength,
    'global_avg': global_avg
}
joblib.dump(encodings, 'models/encodings.pkl')

# Save Models
joblib.dump(score_model, 'models/score_model.pkl')
joblib.dump(win_model, 'models/win_model.pkl')

print("✅ Training Complete! Models and Encodings saved in 'models/' folder.")