import pandas as pd
import numpy as np
import streamlit as st
from xgboost import XGBRegressor
from sklearn.linear_model import LogisticRegression
import random

# ==============================
# 🚀 TRAIN MODELS
# ==============================
@st.cache_resource
def train_models(file_path):
    df = pd.read_csv(file_path)

    df['runs_batter'] = df['runs_batter'].fillna(0)
    df['runs_total'] = df.get('runs_total', df['runs_batter'])
    df['bowler_wicket'] = df['bowler_wicket'].fillna(0)

    df['ball_num'] = (df['over'] * 6) + df['ball']

    # MATCH STATE
    df['cum_runs'] = df.groupby(['match_id','innings'])['runs_total'].cumsum()
    df['cum_wkts'] = df.groupby(['match_id','innings'])['bowler_wicket'].cumsum()
    df['balls'] = df.groupby(['match_id','innings']).cumcount() + 1

    df['overs_left'] = 120 - df['balls']
    df['run_rate'] = df['cum_runs'] / (df['balls']/6 + 0.1)

    # FINAL SCORE TARGET
    final_scores = df.groupby(['match_id','innings'])['runs_total'].sum().reset_index()
    final_scores.rename(columns={'runs_total':'final_score'}, inplace=True)

    df = df.merge(final_scores, on=['match_id','innings'])

    # ================= SCORE MODEL =================
    X_score = df[['cum_runs','cum_wkts','balls','overs_left','run_rate']].fillna(0)
    y_score = df['final_score']

    score_model = XGBRegressor(n_estimators=300, max_depth=8)
    score_model.fit(X_score, y_score)

    # ================= WIN MODEL =================
    df2 = df[df['innings'] == 2].copy()
    df2['target'] = df2.groupby('match_id')['final_score'].transform('first')

    df2['runs_needed'] = df2['target'] - df2['cum_runs']
    df2['balls_left'] = 120 - df2['balls']

    df2['win'] = (df2['runs_needed'] <= 0).astype(int)

    X_win = df2[['runs_needed','balls_left','cum_wkts']].fillna(0)
    y_win = df2['win']

    win_model = LogisticRegression(max_iter=1000)
    win_model.fit(X_win, y_win)

    return score_model, win_model


# ==============================
# 📈 SCORE PREDICTOR (ML)
# ==============================
def predict_score(score_model, runs, wkts, balls):
    overs_left = 120 - balls
    rr = runs / (balls/6 + 0.1)

    X = np.array([[runs, wkts, balls, overs_left, rr]])
    pred = score_model.predict(X)[0]

    return int(pred)


# ==============================
# 🏆 WIN PREDICTOR (ML)
# ==============================
def predict_win(win_model, runs_needed, balls_left, wkts):
    X = np.array([[runs_needed, balls_left, wkts]])
    prob = win_model.predict_proba(X)[0][1]
    return int(prob * 100)


# ==============================
# 🎯 MONTE CARLO SIMULATION
# ==============================
def simulate_ball():
    return random.choices(
        ['dot','1','2','4','6','w'],
        weights=[30,35,10,15,5,5]
    )[0]


def simulate_innings(score, wkts, balls_left):
    s = score
    w = wkts

    for _ in range(balls_left):
        if w >= 10:
            break

        outcome = simulate_ball()

        if outcome == 'w':
            w += 1
        else:
            s += int(outcome)

    return s


def monte_carlo(score, wkts, balls_left, sims=300):
    results = []

    for _ in range(sims):
        results.append(simulate_innings(score, wkts, balls_left))

    return int(np.mean(results)), int(np.percentile(results, 75))


# ==============================
# 🎛️ STREAMLIT UI
# ==============================
st.set_page_config(page_title="IPL AI PRO", layout="wide")
st.title("🏏 IPL AI Engine (ML + Simulation)")

score_model, win_model = train_models("data/IPL.csv")

menu = st.sidebar.radio("Select Tool", [
    "📈 Score Predictor",
    "🏆 Win Predictor"
])

# ==============================
# 📈 SCORE UI
# ==============================
if menu == "📈 Score Predictor":

    st.subheader("📈 ML Score Predictor")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        runs = st.number_input("Current Score", 0, 300, 85)
    with c2:
        wkts = st.slider("Wickets", 0, 10, 2)
    with c3:
        overs = st.number_input("Overs", 0, 20, 10)
    with c4:
        balls = st.number_input("Balls", 0, 5, 0)

    balls_bowled = overs*6 + balls
    balls_left = 120 - balls_bowled

    if st.button("Predict Score"):

        ml_score = predict_score(score_model, runs, wkts, balls_bowled)
        sim_avg, sim_high = monte_carlo(runs, wkts, balls_left)

        st.metric("🤖 ML Prediction", ml_score)
        st.metric("🎲 Simulation Avg", sim_avg)
        st.metric("🚀 High Range", sim_high)


# ==============================
# 🏆 WIN UI
# ==============================
if menu == "🏆 Win Predictor":

    st.subheader("🏆 ML Win Predictor")

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        target = st.number_input("Target", 100, 300, 180)
    with c2:
        score = st.number_input("Score", 0, 300, 75)
    with c3:
        wkts = st.slider("Wickets Lost", 0, 10, 2)
    with c4:
        overs = st.number_input("Overs", 0, 20, 10)
    with c5:
        balls = st.number_input("Balls", 0, 5, 0)

    balls_bowled = overs*6 + balls
    balls_left = 120 - balls_bowled
    runs_needed = target - score

    if st.button("Predict Win"):

        win_prob = predict_win(win_model, runs_needed, balls_left, wkts)

        st.markdown(f"### 🟢 Chase Win Probability: {win_prob}%")
        st.progress(win_prob/100)

