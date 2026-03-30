import pandas as pd
import numpy as np

def prepare_industry_data(file_path):
    df = pd.read_csv(file_path)
    
    df['current_score'] = df.groupby(['match_id', 'innings'])['runs_total'].cumsum()
    
    # Wicket count calculate karna
    df['is_wicket'] = df['player_out'].apply(lambda x: 1 if pd.notnull(x) else 0)
    df['wickets_fallen'] = df.groupby(['match_id', 'innings'])['is_wicket'].cumsum()
    
    # 3. Last 5 overs ka performance (Feature Engineering)
    # Industry models 'context' dekhte hain
    df['runs_last_5_overs'] = df.groupby(['match_id', 'innings'])['runs_total'].rolling(window=30).sum().values
    
    # 4. Target calculate karna (Second Innings ke liye)
    # Kaggle data mein 'runs_target' already hai, but accuracy check zaruri hai
    
    return df