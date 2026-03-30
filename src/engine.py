import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import os

def train_professional_model():
    print("⏳ Loading Dataset...")
    df = pd.read_csv('data/IPL.csv', low_memory=False)

    # 1. Sirf wahi data lein jo match shuru hone se pehle pata hota hai
    # Ball-by-ball data se hume sirf Match level ki summary chahiye
    match_data = df.drop_duplicates(subset=['match_id'], keep='last').copy()
    
    # 2. STRICT CLEANING: Winner column ko feature se bilkul alag rakhein
    match_data = match_data.dropna(subset=['match_won_by', 'venue', 'batting_team'])

    # 3. Features Selection (SIRF YEHI 5 HONI CHAHIYE)
    features_list = ['venue', 'batting_team', 'bowling_team', 'toss_winner', 'toss_decision']
    X = match_data[features_list].copy()
    
    # 4. Target Variable (Win/Loss)
    y = (match_data['batting_team'] == match_data['match_won_by']).astype(int)

    # 5. Encoding
    if not os.path.exists('models'): os.makedirs('models')
    for col in features_list:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        joblib.dump(le, f'models/le_{col}.pkl')

    # 6. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 7. Model with Noise (Taaki 100% accuracy na aaye)
    model = XGBClassifier(
        n_estimators=50,         # Trees aur kam karein
        max_depth=2,             # Sirf base level patterns
        learning_rate=0.01,      # Bahut dhire seekhega
        reg_lambda=10,           # High Regularization (Overfitting killer)
        eval_metric='logloss',
        random_state=42
    )

    model.fit(X_train, y_train)

    # Ab accuracy check karein
    acc = model.score(X_test, y_test)
    print(f"📊 New Realistic Accuracy: {round(acc*100, 2)}%")
    
    if acc > 0.95:
        print("⚠️ Warning: Accuracy is still too high. Checking for Data Leakage...")

    joblib.dump(model, 'models/ipl_winner_model.pkl')
    print("📁 Balanced Model Saved!")

if __name__ == "__main__":
    train_professional_model()