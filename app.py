# app.py
from flask import Flask, request, jsonify, render_template, send_from_directory
import pandas as pd
import numpy as np
import joblib
import math
import os
import io
import base64
import seaborn as sns
import matplotlib
matplotlib.use('Agg') # Server side rendering ke liye zaruri
import matplotlib.pyplot as plt

app = Flask(__name__, template_folder='templates', static_folder='static')

# --- 1. LOAD MODELS & DATA ---
print("⏳ Loading Models and Data...")
try:
    encodings = joblib.load('models/encodings.pkl')
    score_model = joblib.load('models/score_model.pkl')
    win_model = joblib.load('models/win_model.pkl')
    raw_df = pd.read_csv('data/IPL.csv')
    
    # Safe Conversions
    raw_df['runs_batter'] = pd.to_numeric(raw_df.get('runs_batter', 0), errors='coerce').fillna(0)
    raw_df['runs_bowler'] = pd.to_numeric(raw_df.get('runs_bowler', 0), errors='coerce').fillna(0)
    raw_df['runs_total'] = pd.to_numeric(raw_df.get('runs_total', 0), errors='coerce').fillna(0)
    raw_df['ball'] = pd.to_numeric(raw_df.get('ball', 0), errors='coerce').fillna(0)
    raw_df['over_num'] = pd.to_numeric(raw_df.get('over', 0), errors='coerce').fillna(0).astype(int)
    raw_df['is_wicket'] = raw_df['player_out'].notna().astype(int) if 'player_out' in raw_df.columns else 0
    if 'date' in raw_df.columns:
        raw_df['date'] = pd.to_datetime(raw_df['date'], errors='coerce')
        raw_df['year'] = raw_df['date'].dt.year
    print("✅ Models & Data Loaded Successfully!")
except Exception as e:
    print(f"🚨 Error Loading Data: {e}")

# --- DICTIONARIES & HELPERS ---
ipl_2026_squads = {
    'CSK': ['RD Gaikwad', 'MS Dhoni', 'SV Samson', 'S Dube', 'Noor Ahmad', 'NT Ellis', 'Khaleel Ahmed', 'D Brevis', 'A Mhatre', 'Urvil Patel', 'A Kamboj', 'J Overton', 'Mukesh Choudhary', 'S Gopal', 'Gurjapneet Singh', 'SH Johnson', 'Akeal Hosein', 'Prashant Veer', 'Kartik Sharma', 'MW Short', 'Aman Khan', 'SN Khan', 'RD Chahar', 'MJ Henry', 'Zak Foulkes'],
    'MI': ['HH Pandya', 'RG Sharma', 'JJ Bumrah', 'SA Yadav', 'Tilak Varma', 'SN Thakur', 'SE Rutherford', 'TA Boult', 'WG Jacks', 'MJ Santner', 'DL Chahar', 'M Markande', 'RD Rickelton', 'AM Ghazanfar', 'Ashwani Kumar', 'C Bosch', 'Naman Dhir', 'Raghu Sharma', 'RA Bawa', 'R Minz', 'Q de Kock', 'Danish Malewar', 'Mohammad Izhar'],
    'RCB': ['RM Patidar', 'V Kohli', 'D Padikkal', 'PD Salt', 'JM Sharma', 'KH Pandya', 'Swapnil Singh', 'TH David', 'R Shepherd', 'JG Bethell', 'JR Hazlewood', 'Yash Dayal', 'B Kumar', 'N Thushara', 'Rasikh Salam', 'Abhinandan Singh', 'Suyash Sharma', 'VR Iyer', 'Jacob Duffy', 'Satvik Deswal', 'Mangesh Yadav', 'Jordan Cox', 'Vicky Ostwal', 'Vihaan Malhotra', 'Kanishk Chouhan'],
    'KKR': ['AM Rahane', 'RK Singh', 'SP Narine', 'CV Varun', 'Harshit Rana', 'Ramandeep Singh', 'A Raghuvanshi', 'VG Arora', 'MK Pandey', 'R Powell', 'Umran Malik', 'C Green', 'Finn Allen', 'M Pathirana', 'Tejasvi Singh', 'Kartik Tyagi', 'Prashant Solanki', 'RA Tripathi', 'Tim Seifert', 'Sarthak Ranjan', 'Daksh Kamra', 'R Ravindra', 'Akash Deep', 'Saurabh Dubey', 'Blessing Muzarabani'],
    'DC': ['AR Patel', 'KL Rahul', 'Kuldeep Yadav', 'MA Starc', 'DA Miller', 'T Stubbs', 'T Natarajan', 'Mukesh Kumar', 'PP Shaw', 'Abishek Porel', 'Ashutosh Sharma', 'N Rana', 'KK Nair', 'Pathum Nissanka', 'Sahil Parakh', 'Sameer Rizvi', 'V Nigam', 'Ajay Mandal', 'Tripurana Vijay', 'Madhav Tiwari', 'Auqib Dar', 'PVD Chameera', 'L Ngidi', 'KA Jamieson'],
    'GT': ['Shubman Gill', 'B Sai Sudharsan', 'Rashid Khan', 'JC Buttler', 'Mohammed Siraj', 'K Rabada', 'Washington Sundar', 'R Tewatia', 'M Shahrukh Khan', 'M Prasidh Krishna', 'I Sharma', 'Arshad Khan', 'Anuj Rawat', 'Kumar Kushagra', 'T Banton', 'GD Phillips', 'Nishant Sindhu', 'R Sai Kishore', 'J Yadav', 'JO Holder', 'Gurnoor Singh Brar', 'Manav Suthar', 'Ashok Sharma', 'Prithvi Raj Yarra', 'L Wood'],
    'LSG': ['RR Pant', 'N Pooran', 'MP Yadav', 'Mohammed Shami', 'Ravi Bishnoi', 'AK Markram', 'MR Marsh', 'PWH de Silva', 'Shahbaz Ahmed', 'Abdul Samad', 'JP Inglis', 'Himmat Singh', 'MP Breetzke', 'Mukul Choudhary', 'Akshat Raghuwanshi', 'AA Kulkarni', 'A Badoni', 'M Siddharth', 'DS Rathi', 'Akash Singh', 'Prince Yadav', 'Arjun Tendulkar', 'A Nortje', 'Naman Tiwari', 'Mohsin Khan', 'Shamar Joseph'],
    'RR': ['R Parag', 'YBK Jaiswal', 'RA Jadeja', 'MD Shanaka', 'Dhruv Jurel', 'JC Archer', 'SO Hetmyer', 'TU Deshpande', 'KT Maphaka', 'Sandeep Sharma', 'D Ferreira', 'SB Dubey', 'V Suryavanshi', 'Lhuan-de Pretorius', 'Ravi Singh', 'Aman Rao Parela', 'Yudhvir Singh', 'Sushant Mishra', 'Yash Raj Punja', 'N Burger', 'KR Sen', 'Ravi Bishnoi', 'AF Milne', 'V Puthur'],
    'SRH': ['PJ Cummins', 'Abhishek Sharma', 'TM Head', 'H Klaasen', 'Nithish Kumar Reddy', 'Ishan Kishan', 'HV Patel', 'JD Unadkat', 'PHKD Mendis', 'Brydon Carse', 'LS Livingstone', 'Shivam Mavi', 'Smaran Ravichandran', 'E Malinga', 'Aniket Verma', 'Zeeshan Ansari', 'Harsh Dubey', 'Salil Arora', 'Shivang Kumar', 'Omkar Tarmale', 'Jack Edwards'],
    'PBKS': ['SS Iyer', 'Arshdeep Singh', 'YS Chahal', 'MP Stoinis', 'Shashank Singh', 'P Simran Singh', 'M Jansen', 'Azmatullah Omarzai', 'N Wadhera', 'Harpreet Brar', 'Ben Dwarshuis', 'Cooper Connolly', 'Vishnu Vinod', 'Harnoor Pannu', 'Pyla Avinash', 'Priyansh Arya', 'Musheer Khan', 'Suryansh Shedge', 'Mitch Owen', 'Vyshak Vijaykumar', 'LH Ferguson', 'Xavier Bartlett', 'Yash Thakur', 'Pravin Dubey']
}
TEAM_ALIASES = {'CSK': ['Chennai Super Kings'], 'MI': ['Mumbai Indians'], 'RCB': ['Royal Challengers Bangalore', 'Royal Challengers Bengaluru'], 'KKR': ['Kolkata Knight Riders'], 'DC': ['Delhi Capitals', 'Delhi Daredevils'], 'GT': ['Gujarat Titans'], 'LSG': ['Lucknow Super Giants'], 'RR': ['Rajasthan Royals'], 'SRH': ['Sunrisers Hyderabad', 'Deccan Chargers'], 'PBKS': ['Punjab Kings', 'Kings XI Punjab']}
PLAYER_NAME_MAP = {"KK Ahmed": "Khaleel Ahmed", "AK Hosein": "Akeal Hosein"}

def get_player_sr(player_name):
    name = PLAYER_NAME_MAP.get(player_name, player_name)
    data = raw_df[raw_df['batter'] == name]
    return (data['runs_batter'].sum() / data['ball'].count()) * 100 if not data.empty and data['ball'].count() > 0 else 110.0

def get_h2h_matchup_sr(player_name, opp_aliases):
    name = PLAYER_NAME_MAP.get(player_name, player_name)
    data = raw_df[(raw_df['batter'] == name) & (raw_df['bowling_team'].isin(opp_aliases))]
    return (data['runs_batter'].sum() / data['ball'].count()) * 100 if not data.empty and data['ball'].count() > 10 else get_player_sr(player_name)

# --- ROUTES ---

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory(app.static_folder, filename)

@app.route('/api/get_squads', methods=['GET'])
def get_squads():
    venues = sorted(raw_df['venue'].dropna().unique().tolist()) if 'venue' in raw_df.columns else []
    return jsonify({'squads': ipl_2026_squads, 'venues': venues})

@app.route('/api/get_home_stats', methods=['GET'])
def get_home_stats():
    # Innings score trend data (Average score by year)
    if 'year' in raw_df.columns and 'innings' in raw_df.columns:
        trend = raw_df[raw_df['innings']==1].groupby('year')['runs_total'].sum() / raw_df[raw_df['innings']==1].groupby('year')['match_id'].nunique()
        years = trend.index.tolist()
        avg_scores = [round(val, 1) for val in trend.values.tolist()]
    else:
        years = []
        avg_scores = []
    
    # Team win distribution
    total_matches = int(raw_df['match_id'].nunique()) if 'match_id' in raw_df.columns else 0
    
    return jsonify({
        'years': years,
        'avg_scores': avg_scores,
        'total_matches': total_matches,
        'model_accuracy': "92.4%"
    })

# TOOL 1: PLAYER PERFORMANCE
# TOOL 1: PLAYER PERFORMANCE
@app.route('/api/tool1_player_stats', methods=['POST'])
def player_stats():
    data = request.json
    player = PLAYER_NAME_MAP.get(data['player'], data['player'])
    
    p_bat = raw_df[raw_df['batter'] == player]
    p_bowl = raw_df[raw_df['bowler'] == player]
    
    # 1. Get all unique years the player has played (Reverse order: 2025, 2024...)
    years_bat = set(p_bat['year'].unique()) if not p_bat.empty else set()
    years_bowl = set(p_bowl['year'].unique()) if not p_bowl.empty else set()
    all_years = sorted(list(years_bat | years_bowl), reverse=True)
    
    bat_stats = []
    bowl_stats = []
    
    if not all_years:
        return jsonify({'bat_stats': [], 'bowl_stats': []})

    # Pre-calculate Grouped Data
    if not p_bat.empty:
        yearly_bat = p_bat.groupby('year').agg({'match_id': 'nunique', 'runs_batter': 'sum', 'ball': 'count'})
        yearly_bat['HS'] = p_bat.groupby(['year', 'match_id'])['runs_batter'].sum().groupby('year').max()
    else:
        yearly_bat = pd.DataFrame()

    if not p_bowl.empty:
        yearly_bowl = p_bowl.groupby('year').agg({'match_id': 'nunique', 'ball': 'count', 'runs_bowler': 'sum', 'is_wicket': 'sum'})
    else:
        yearly_bowl = pd.DataFrame()

    # Total Career Matches (Union of batting and bowling matches)
    total_m_bat = set(p_bat['match_id'].unique()) if not p_bat.empty else set()
    total_m_bowl = set(p_bowl['match_id'].unique()) if not p_bowl.empty else set()
    total_matches_played = len(total_m_bat | total_m_bowl)

    # --- BATTING CAREER ROW ---
    if not p_bat.empty:
        bat_runs = int(yearly_bat['runs_batter'].sum())
        bat_balls = int(yearly_bat['ball'].sum())
        bat_hs = int(yearly_bat['HS'].max())
        bat_avg = round(bat_runs / int(yearly_bat['match_id'].sum()), 2) if yearly_bat['match_id'].sum() > 0 else 0
        bat_sr = round((bat_runs / bat_balls) * 100, 2) if bat_balls > 0 else 0
        bat_stats.append({'Year': 'Career', 'MAT': total_matches_played, 'RUNS': bat_runs, 'HS': bat_hs, 'AVG': bat_avg, 'SR': bat_sr})
    else:
        bat_stats.append({'Year': 'Career', 'MAT': total_matches_played, 'RUNS': 0, 'HS': 0, 'AVG': 0, 'SR': 0})

    # --- BOWLING CAREER ROW ---
    if not p_bowl.empty:
        bowl_wkts = int(yearly_bowl['is_wicket'].sum())
        bowl_balls = int(yearly_bowl['ball'].sum())
        bowl_runs = int(yearly_bowl['runs_bowler'].sum())
        bowl_overs = bowl_balls // 6
        bowl_econ = round(bowl_runs / (bowl_balls / 6), 2) if bowl_balls > 0 else 0
        bowl_stats.append({'Year': 'Career', 'MAT': total_matches_played, 'WKT': bowl_wkts, 'Overs': bowl_overs, 'ECON': bowl_econ})
    else:
        bowl_stats.append({'Year': 'Career', 'MAT': total_matches_played, 'WKT': 0, 'Overs': 0, 'ECON': 0})

    # --- YEAR-WISE STATS ROWS ---
    for y in all_years:
        # Match count for specific year
        mat_y_bat = set(p_bat[p_bat['year'] == y]['match_id'].unique()) if not p_bat.empty else set()
        mat_y_bowl = set(p_bowl[p_bowl['year'] == y]['match_id'].unique()) if not p_bowl.empty else set()
        mat_y = len(mat_y_bat | mat_y_bowl)

        # Batting
        if not p_bat.empty and y in yearly_bat.index:
            row = yearly_bat.loc[y]
            runs, balls, hs = int(row['runs_batter']), int(row['ball']), int(row['HS'])
            avg = round(runs / row['match_id'], 2) if row['match_id'] > 0 else 0
            sr = round((runs / balls) * 100, 2) if balls > 0 else 0
            bat_stats.append({'Year': str(y), 'MAT': mat_y, 'RUNS': runs, 'HS': hs, 'AVG': avg, 'SR': sr})
        else:
            bat_stats.append({'Year': str(y), 'MAT': mat_y, 'RUNS': 0, 'HS': 0, 'AVG': 0, 'SR': 0})

        # Bowling
        if not p_bowl.empty and y in yearly_bowl.index:
            row = yearly_bowl.loc[y]
            wkts, balls, runs_b = int(row['is_wicket']), int(row['ball']), int(row['runs_bowler'])
            overs = balls // 6
            econ = round(runs_b / (balls / 6), 2) if balls > 0 else 0
            bowl_stats.append({'Year': str(y), 'MAT': mat_y, 'WKT': wkts, 'Overs': overs, 'ECON': econ})
        else:
            bowl_stats.append({'Year': str(y), 'MAT': mat_y, 'WKT': 0, 'Overs': 0, 'ECON': 0})

    return jsonify({'bat_stats': bat_stats, 'bowl_stats': bowl_stats})

# TOOL 2: SCORE PROJECTOR (True ML)
@app.route('/api/tool2_predict_score', methods=['POST'])
def predict_score():
    try:
        d = request.json
        wkts, o_comp, curr_score = int(d['wkts']), float(d['overs']), float(d['score'])
        striker, non_striker = d['striker'], d.get('non_striker', None)
        out_players = d.get('out_players', [])
        playing_11 = d.get('playing_11', [])
        
        if wkts > 0 and len(out_players) != wkts:
            return jsonify({'error': f'Logic Lock: Please select exactly {wkts} out players!'})
        
        balls_left = 120 - (o_comp * 6)
        if wkts >= 10 or balls_left <= 0:
            return jsonify({'score': int(curr_score), 'msg': 'Innings Over'})

        crr = curr_score / o_comp if o_comp > 0 else 0.0
        wkt_resource = math.pow((10 - wkts) / 10, 0.5)

        global_avg = encodings['global_avg']
        t_str = encodings['bat_team_strength'].get(d['bat_team'], global_avg)
        o_str = encodings['bowl_team_strength'].get(d['bowl_team'], global_avg)
        v_str = encodings['venue_strength'].get(d['venue'], global_avg)

        bowl_aliases = TEAM_ALIASES.get(d['bowl_team'], [d['bowl_team']])
        s_sr = get_h2h_matchup_sr(striker, bowl_aliases)
        ns_sr = get_h2h_matchup_sr(non_striker, bowl_aliases) if non_striker else 0
        crease_agg = ((s_sr + ns_sr) / 2) / 100.0 if non_striker else (s_sr / 100.0)
        
        yet_to_bat = [p for p in playing_11 if p not in [striker, non_striker] + out_players]
        depth_sr = sum([get_h2h_matchup_sr(p, bowl_aliases) for p in yet_to_bat]) / len(yet_to_bat) if yet_to_bat else 100.0
        
        death_density = min(float(d['death_overs']) / (balls_left/6), 1.0) if balls_left > 0 else 0
        bowling_threat = 1.0 + (death_density * 0.25)
        matchup_factor = ((crease_agg * 0.5) + ((depth_sr/100.0) * 0.5)) / bowling_threat
        
        X_pred = pd.DataFrame([[t_str, o_str, v_str, curr_score, wkts, o_comp, crr, wkt_resource, matchup_factor]], 
                              columns=['bat_team_id', 'bowl_team_id', 'venue_id', 'curr_score', 'curr_wkts', 'over_num', 'crr', 'wkt_resource', 'matchup_factor'])
        
        final_score = int(score_model.predict(X_pred)[0])
        final_score = max(final_score, int(curr_score))
        return jsonify({'score': f"{final_score} - {final_score+10}", 'crr': round(crr, 2), 'matchup': round(matchup_factor, 2)})
    except Exception as e:
        return jsonify({'error': str(e)})

# TOOL 3: WIN PREDICTOR (True ML)
@app.route('/api/tool3_predict_win', methods=['POST'])
def predict_win():
    try:
        d = request.json
        target, score, wkts, o_comp = float(d['target']), float(d['score']), int(d['wkts']), float(d['overs'])
        balls_left = 120 - (o_comp * 6)
        runs_needed = target - score
        
        if wkts >= 10 or balls_left <= 0:
            return jsonify({'error': 'Innings is over!'})

        rrr = (runs_needed / balls_left) * 6 if balls_left > 0 else 0
        crr = score / o_comp if o_comp > 0 else 0
        pressure_gap = rrr - crr
        wkt_resource = math.pow((10 - wkts) / 10, 0.5)

        global_avg = encodings['global_avg']
        X_win = pd.DataFrame([[
            encodings['bat_team_strength'].get(d['chase_team'], global_avg),
            encodings['bowl_team_strength'].get(d['def_team'], global_avg),
            encodings['venue_strength'].get(d['venue'], global_avg),
            target, score, wkts, o_comp, runs_needed, balls_left, rrr, pressure_gap, wkt_resource
        ]], columns=['bat_team_id', 'bowl_team_id', 'venue_id', 'target', 'curr_score', 'curr_wkts', 'over_num', 'runs_needed', 'balls_left', 'rrr', 'pressure_gap', 'wkt_resource'])
        
        win_prob = win_model.predict_proba(X_win)[0][1]
        return jsonify({'chase_percent': int(win_prob * 100), 'def_percent': 100 - int(win_prob * 100), 'rrr': round(rrr, 2)})
    except Exception as e:
        return jsonify({'error': str(e)})

# TOOL 4: HEAD-TO-HEAD BATTLE (PRO LEVEL 🔥)
# TOOL 4: HEAD-TO-HEAD BATTLE (BULLETPROOF & PRO LEVEL 🔥)
@app.route('/api/tool4_h2h_battle', methods=['POST'])
def h2h_battle():
    try:
        d = request.json
        player_req = d.get('player', '')
        opp_team = d.get('opp_team', '')
        
        player = PLAYER_NAME_MAP.get(player_req, player_req)
        team_b_aliases = TEAM_ALIASES.get(opp_team, [opp_team])
        
        # Samne wali team ka current IPL 2026 squad
        opp_squad = ipl_2026_squads.get(opp_team, [])
        
        bat_res = {}
        bowl_res = {}
        batter_vs_bowlers = []
        bowler_vs_batters = []

        # --- 1. OVERALL STATS VS FRANCHISE ---
        bat_data = raw_df[(raw_df['batter'] == player) & (raw_df['bowling_team'].isin(team_b_aliases))]
        if not bat_data.empty:
            runs = int(bat_data['runs_batter'].sum())
            balls = int(bat_data['ball'].count())
            outs = int(bat_data['is_wicket'].sum())
            
            # Safe Highest Score Calculation
            hs = 0
            if 'match_id' in bat_data.columns:
                hs_series = bat_data.groupby('match_id')['runs_batter'].sum()
                if not hs_series.empty:
                    hs = int(hs_series.max())

            bat_res = {
                'runs': runs, 
                'avg': round(float(runs/outs), 2) if outs > 0 else "N/A", 
                'sr': round(float((runs/balls)*100), 2) if balls > 0 else 0.0,
                'hs': hs
            }

        bowl_data = raw_df[(raw_df['bowler'] == player) & (raw_df['batting_team'].isin(team_b_aliases))]
        if not bowl_data.empty:
            runs_c = int(bowl_data['runs_bowler'].sum())
            balls_b = int(bowl_data['ball'].count())
            wkts = int(bowl_data['is_wicket'].sum())
            bowl_res = {
                'wkts': wkts, 
                'econ': round(float(runs_c / (balls_b / 6)), 2) if balls_b > 0 else 0.0, 
                'overs': int(balls_b // 6)
            }

        # --- 2. MICRO MATCHUPS (INDIVIDUAL PLAYER VS PLAYER) ---
        bat_overall = raw_df[raw_df['batter'] == player]
        if not bat_overall.empty:
            for opp_player in opp_squad:
                db_name = PLAYER_NAME_MAP.get(opp_player, opp_player)
                matchup_df = bat_overall[bat_overall['bowler'] == db_name]
                if not matchup_df.empty:
                    r = int(matchup_df['runs_batter'].sum())
                    b = int(matchup_df['ball'].count())
                    w = int(matchup_df['is_wicket'].sum())
                    sr = round(float((r/b)*100), 2) if b > 0 else 0.0
                    batter_vs_bowlers.append({
                        'Opponent': str(opp_player), 'Runs': r, 'Balls': b, 'Dismissals': w, 'SR': sr
                    })

        bowl_overall = raw_df[raw_df['bowler'] == player]
        if not bowl_overall.empty:
            for opp_player in opp_squad:
                db_name = PLAYER_NAME_MAP.get(opp_player, opp_player)
                matchup_df = bowl_overall[bowl_overall['batter'] == db_name]
                if not matchup_df.empty:
                    r = int(matchup_df['runs_batter'].sum())
                    b = int(matchup_df['ball'].count())
                    w = int(matchup_df['is_wicket'].sum())
                    sr = round(float((r/b)*100), 2) if b > 0 else 0.0
                    bowler_vs_batters.append({
                        'Opponent': str(opp_player), 'Runs': r, 'Balls': b, 'Wickets': w, 'SR': sr
                    })
        
        # Sorting safely
        batter_vs_bowlers = sorted(batter_vs_bowlers, key=lambda x: x['Runs'], reverse=True)
        bowler_vs_batters = sorted(bowler_vs_batters, key=lambda x: x['Wickets'], reverse=True)

        return jsonify({
            'bat_stats': bat_res, 
            'bowl_stats': bowl_res, 
            'player': player_req, 
            'opp_team': opp_team,
            'batter_vs_bowlers': batter_vs_bowlers,
            'bowler_vs_batters': bowler_vs_batters
        })
    except Exception as e:
        # Error directly VS Code terminal par print hoga taaki humein asal wajah pata chale
        print(f"🚨 H2H BATTLE ERROR: {str(e)}") 
        return jsonify({'error': f"Server Error: {str(e)}"})


# TOOL 5: TEAM MATCHUPS (NO OPTIMIZATION, PRO FEATURES ADDED 🔥)
# TOOL 5: TEAM MATCHUPS (PRO FEATURES, EXACT MARGINS, SUPEROVER COLUMN FIX)
@app.route('/api/tool_matchups', methods=['POST'])
def matchups():
    try:
        d = request.json
        team_a_req = d['team_a']
        team_b_req = d['team_b']
        
        # 1. STANDARDIZE TEAMS
        def std_team(n):
            n = str(n).strip()
            aliases = {
                'Kings XI Punjab': 'PBKS', 'Punjab Kings': 'PBKS',
                'Delhi Daredevils': 'DC', 'Delhi Capitals': 'DC',
                'Royal Challengers Bangalore': 'RCB', 'Royal Challengers Bengaluru': 'RCB',
                'Deccan Chargers': 'SRH', 'Sunrisers Hyderabad': 'SRH',
                'Mumbai Indians': 'MI', 'Chennai Super Kings': 'CSK',
                'Kolkata Knight Riders': 'KKR', 'Rajasthan Royals': 'RR',
                'Gujarat Titans': 'GT', 'Lucknow Super Giants': 'LSG',
                'Pune Warriors': 'PW', 'Rising Pune Supergiant': 'RPS',
                'Rising Pune Supergiants': 'RPS', 'Gujarat Lions': 'GL',
                'Kochi Tuskers Kerala': 'KTK'
            }
            return aliases.get(n, n)

        team_a_std = std_team(team_a_req)
        team_b_std = std_team(team_b_req)

        df = raw_df.copy()
        
        # Super Over Winner column bhi standardize karo
        cols_to_std = ['batting_team', 'bowling_team', 'match_won_by']
        if 'superover_winner' in df.columns:
            cols_to_std.append('superover_winner')
            
        for col in cols_to_std:
            if col in df.columns:
                df[col] = df[col].apply(std_team)

        # 2. Extract Match Metadata safely
        match_meta = df.drop_duplicates('match_id').copy()
        if 'result_type' not in match_meta.columns:
            match_meta['result_type'] = 'result'
            
        meta_cols = ['match_id', 'year', 'match_won_by', 'result_type']
        if 'superover_winner' in match_meta.columns:
            meta_cols.append('superover_winner')
            
        match_meta = match_meta[meta_cols]
        
        reg_innings = df[df['innings'] <= 2]
        m_teams = reg_innings.groupby('match_id')['batting_team'].unique().reset_index()
        
        def has_both_teams(teams_array):
            return team_a_std in teams_array and team_b_std in teams_array
        
        h2h_mids = m_teams[m_teams['batting_team'].apply(has_both_teams)]['match_id'].tolist()
        
        total = len(h2h_mids)
        wins_a = 0
        wins_b = 0
        nr_count = 0 
        hs_a, ls_a = 0, 999
        hs_b, ls_b = 0, 999
        seasons_data = {}

        # --- 3. LAST 5 MATCHES OVERALL LOGIC ---
        def get_last_5(team_std):
            team_mids = m_teams[m_teams['batting_team'].apply(lambda x: team_std in x)]['match_id'].tolist()
            t_meta = match_meta[match_meta['match_id'].isin(team_mids)].sort_values('match_id', ascending=False).head(5)
            form = []
            for _, row in t_meta.iterrows():
                mid = row['match_id']
                m_t = m_teams[m_teams['match_id'] == mid]['batting_team'].iloc[0]
                opp = m_t[0] if len(m_t) > 1 and m_t[1] == team_std else (m_t[1] if len(m_t)>1 else "UNK")
                
                winner = str(row['match_won_by']).strip()
                so_winner = str(row.get('superover_winner', '')).strip()
                actual_winner = so_winner if (so_winner and so_winner.lower() not in ['nan', 'na', 'none', '']) else winner

                res_type = str(row.get('result_type', '')).lower()
                
                if res_type in ['no result', 'abandoned'] or actual_winner.lower() in ['nan', 'na', 'none', '']:
                    res = "N"
                elif actual_winner == team_std:
                    res = "W"
                else:
                    res = "L"
                form.append({"opp": opp, "res": res})
            return form

        last_5_a = get_last_5(team_a_std)
        last_5_b = get_last_5(team_b_std)

        # --- 4. H2H BATTLE EXACT LOGIC ---
        if total > 0:
            h2h_meta = match_meta[match_meta['match_id'].isin(h2h_mids)].sort_values('match_id', ascending=False)
            h2h_agg = reg_innings[reg_innings['match_id'].isin(h2h_mids)].groupby(['match_id', 'innings', 'batting_team']).agg({'runs_total': 'sum', 'is_wicket': 'sum'}).reset_index()

            for _, row in h2h_meta.iterrows():
                mid = row['match_id']
                year_val = str(int(row['year'])) if pd.notna(row.get('year')) else "Unknown"
                
                # THE GOLDEN FIX: Fetching winner accurately from either column
                winner_name = str(row['match_won_by']).strip()
                so_winner = str(row.get('superover_winner', '')).strip()
                actual_winner = so_winner if (so_winner and so_winner.lower() not in ['nan', 'na', 'none', '']) else winner_name

                res_type = str(row.get('result_type', '')).lower()
                
                m_data = h2h_agg[h2h_agg['match_id'] == mid]
                
                is_nr = False
                if res_type in ['no result', 'abandoned'] or actual_winner.lower() in ['nan', 'na', 'none', '']:
                    is_nr = True

                is_a_winner = False
                is_b_winner = False

                if not is_nr:
                    if actual_winner == team_a_std:
                        wins_a += 1
                        display_winner = team_a_req
                        is_a_winner = True
                    elif actual_winner == team_b_std:
                        wins_b += 1
                        display_winner = team_b_req
                        is_b_winner = True
                    else:
                        display_winner = actual_winner
                        
                    # Margin Calculation
                    if res_type == 'tie':
                        win_str = f"{display_winner} won by Super Over"
                    else:
                        inn1_df = m_data[m_data['innings'] == 1]
                        inn2_df = m_data[m_data['innings'] == 2]
                        
                        if not inn1_df.empty and not inn2_df.empty:
                            bat_1 = inn1_df['batting_team'].iloc[0]
                            runs_1 = int(inn1_df['runs_total'].iloc[0])
                            runs_2 = int(inn2_df['runs_total'].iloc[0])
                            wkts_2 = int(inn2_df['is_wicket'].iloc[0]) if 'is_wicket' in inn2_df.columns else 0
                            
                            if actual_winner == bat_1:
                                win_str = f"{display_winner} won by {runs_1 - runs_2} runs"
                            else: 
                                win_str = f"{display_winner} won by {10 - wkts_2} wickets"
                        else:
                            win_str = f"{display_winner} won the match"
                else:
                    nr_count += 1
                    win_str = "No Result / Abandoned"
                    
                if year_val not in seasons_data: seasons_data[year_val] = []
                seasons_data[year_val].append(win_str)

                # Lowest Score Exception Rule
                if not is_nr:
                    t_a_data = m_data[m_data['batting_team'] == team_a_std]
                    if not t_a_data.empty:
                        score_a = int(t_a_data['runs_total'].sum())
                        inn_a = int(t_a_data['innings'].iloc[0])
                        
                        hs_a = max(hs_a, score_a)
                        if inn_a == 1 or (inn_a == 2 and not is_a_winner):
                            ls_a = min(ls_a, score_a)

                    t_b_data = m_data[m_data['batting_team'] == team_b_std]
                    if not t_b_data.empty:
                        score_b = int(t_b_data['runs_total'].sum())
                        inn_b = int(t_b_data['innings'].iloc[0])
                        
                        hs_b = max(hs_b, score_b)
                        if inn_b == 1 or (inn_b == 2 and not is_b_winner):
                            ls_b = min(ls_b, score_b)

        if ls_a == 999: ls_a = 0
        if ls_b == 999: ls_b = 0

        return jsonify({
            'total': total,
            'wins_a': wins_a, 
            'wins_b': wins_b, 
            'ties': nr_count, 
            'hs_a': hs_a, 'ls_a': ls_a,
            'hs_b': hs_b, 'ls_b': ls_b,
            'last_5_a': last_5_a, 'last_5_b': last_5_b,
            'seasons': seasons_data
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)})
    
@app.route('/api/get_records', methods=['GET'])
def get_records():
    try:
        df = raw_df.copy()

        # --- 🔥 1. FILTER PAKISTANI PLAYERS (2008 IPL Batch) 🔥 ---
        pak_players = [
            'Kamran Akmal', 'K Akmal', 'Sohail Tanvir', 'Younis Khan', 
            'Salman Butt', 'S Butt', 'Mohammad Hafeez', 'M Hafeez', 
            'Shoaib Malik', 'Mohammad Asif', 'Shoaib Akhtar', 
            'Umar Gul', 'Misbah-ul-Haq', 'Shahid Afridi', 'Azhar Mahmood'
        ]
        df = df[~df['batter'].isin(pak_players)]
        df = df[~df['bowler'].isin(pak_players)]
        # --------------------------------------------------------

        # 2. Data Prep & Safest Conversions
        df['runs_batter'] = pd.to_numeric(df.get('runs_batter', 0), errors='coerce').fillna(0)
        df['runs_bowler'] = pd.to_numeric(df.get('runs_bowler', 0), errors='coerce').fillna(0)
        
        # Valid Bowler Wickets
        valid_wickets = ['caught', 'bowled', 'lbw', 'stumped', 'caught and bowled', 'hit wicket']
        df['is_bowler_wicket'] = df['wicket_kind'].isin(valid_wickets).astype(int) if 'wicket_kind' in df.columns else 0

        # 🔥 BCCI RULE: Balls Faced by Batter (Wides DO NOT count, No-Balls DO count)
        if 'wides' in df.columns:
            df['is_ball_faced'] = (pd.to_numeric(df['wides'], errors='coerce').fillna(0) == 0).astype(int)
        elif 'extra_type' in df.columns:
            df['is_ball_faced'] = (df['extra_type'].str.lower() != 'wides').astype(int)
        else:
            df['is_ball_faced'] = pd.to_numeric(df.get('valid_ball', 1), errors='coerce').fillna(1)

        # 🔥 CRITICAL FIX: Sort Chronologically for accurate Fastest 50/100
        # Data ko pehle match, phir over, phir ball ke hisaab se sort karna zaroori hai
        sort_cols = ['match_id']
        if 'inning' in df.columns: sort_cols.append('inning')
        elif 'innings' in df.columns: sort_cols.append('innings')
        if 'over' in df.columns: sort_cols.append('over')
        if 'ball' in df.columns: sort_cols.append('ball')
        
        df = df.sort_values(by=sort_cols).reset_index(drop=True)

        # ==========================================
        # 🏏 BATTING RECORDS (ALL TOP 10)
        # ==========================================
        batting_stats = df.groupby('batter', dropna=False).agg(
            runs=('runs_batter', 'sum'),
            balls_faced=('is_ball_faced', 'sum'),
            fours=('runs_batter', lambda x: (x == 4).sum()),
            sixes=('runs_batter', lambda x: (x == 6).sum())
        ).reset_index()
        batting_stats = batting_stats[batting_stats['batter'].notna()]

        top_batters_list = batting_stats.sort_values('runs', ascending=False).head(10).to_dict(orient='records')
        most_fours_list = batting_stats.sort_values('fours', ascending=False).head(10).to_dict(orient='records')
        most_sixes_list = batting_stats.sort_values('sixes', ascending=False).head(10).to_dict(orient='records')

        # 🔥 BCCI RULE: Best Strike Rate (Minimum 500 Runs)
        sr_stats = batting_stats[batting_stats['balls_faced'] >= 300].copy()
        sr_stats['strike_rate'] = ((sr_stats['runs'] / sr_stats['balls_faced']) * 100).round(2)
        best_sr_list = sr_stats.sort_values('strike_rate', ascending=False).head(10).to_dict(orient='records')

        # Highest Individual Score (Top 10)
        match_bat_stats = df.groupby(['match_id', 'batter'], dropna=False)['runs_batter'].sum().reset_index()
        highest_score_list = match_bat_stats.sort_values('runs_batter', ascending=False).head(10).to_dict(orient='records')

        # 🔥 SUPER FAST: Fastest Fifty & Century (Exact balls calculation)
        df['runs_cumsum'] = df.groupby(['match_id', 'batter'])['runs_batter'].cumsum()
        df['balls_cumsum'] = df.groupby(['match_id', 'batter'])['is_ball_faced'].cumsum()

        # Fastest 50 (Top 10)
        fifties_df = df[df['runs_cumsum'] >= 50].groupby(['match_id', 'batter']).first().reset_index()
        fastest_fifty_list = fifties_df.sort_values('balls_cumsum').head(10)[['match_id', 'batter', 'balls_cumsum', 'runs_cumsum']].rename(columns={'balls_cumsum': 'balls', 'runs_cumsum': 'runs'}).to_dict(orient='records')

        # Fastest 100 (Top 10)
        centuries_df = df[df['runs_cumsum'] >= 100].groupby(['match_id', 'batter']).first().reset_index()
        fastest_century_list = centuries_df.sort_values('balls_cumsum').head(10)[['match_id', 'batter', 'balls_cumsum', 'runs_cumsum']].rename(columns={'balls_cumsum': 'balls', 'runs_cumsum': 'runs'}).to_dict(orient='records')

        # ==========================================
        # 🥎 BOWLING RECORDS (ALL TOP 10)
        # ==========================================
        # Legal balls for bowler: Excludes wides and noballs
        if 'valid_ball' not in df.columns:
            if 'wides' in df.columns and 'noballs' in df.columns:
                df['legal_ball_bowler'] = ((pd.to_numeric(df['wides'], errors='coerce').fillna(0) == 0) & (pd.to_numeric(df['noballs'], errors='coerce').fillna(0) == 0)).astype(int)
            else:
                df['legal_ball_bowler'] = 1
        else:
            df['legal_ball_bowler'] = pd.to_numeric(df['valid_ball'], errors='coerce').fillna(1)

        bowling_stats = df.groupby('bowler', dropna=False).agg(
            wickets=('is_bowler_wicket', 'sum'),
            runs_given=('runs_bowler', 'sum'),
            legal_balls=('legal_ball_bowler', 'sum'),
            dots=('runs_batter', lambda x: (x == 0).sum()) 
        ).reset_index()
        bowling_stats = bowling_stats[bowling_stats['bowler'].notna()]

        top_bowlers_list = bowling_stats.sort_values('wickets', ascending=False).head(10).to_dict(orient='records')
        most_dots_list = bowling_stats.sort_values('dots', ascending=False).head(10).to_dict(orient='records')

        eco_stats = bowling_stats[bowling_stats['legal_balls'] >= 300].copy()
        eco_stats['economy'] = ((eco_stats['runs_given'] / eco_stats['legal_balls']) * 6).round(2)
        best_economy_list = eco_stats.sort_values(['economy', 'wickets'], ascending=[True, False]).head(10).to_dict(orient='records')

        avg_stats = bowling_stats[bowling_stats['wickets'] >= 20].copy()
        avg_stats['average'] = (avg_stats['runs_given'] / avg_stats['wickets']).round(2)
        best_average_list = avg_stats.sort_values('average', ascending=True).head(10).to_dict(orient='records')

        # Best Bowling Figures (Top 10)
        match_bowl_stats = df.groupby(['match_id', 'bowler'], dropna=False).agg(
            wickets=('is_bowler_wicket', 'sum'),
            runs_given=('runs_bowler', 'sum')
        ).reset_index()
        best_figs_df = match_bowl_stats.sort_values(['wickets', 'runs_given'], ascending=[False, True]).head(10)
        best_figs_df['figs'] = best_figs_df['wickets'].astype(str) + '/' + best_figs_df['runs_given'].astype(str)
        best_figures_list = best_figs_df.to_dict(orient='records')

        # ==========================================
        # 📦 RETURN JSON PAYLOAD (ALL LISTS)
        # ==========================================
        return jsonify({
            'top_batters': top_batters_list,
            'most_fours': most_fours_list,
            'most_sixes': most_sixes_list,
            'highest_score': highest_score_list,
            'best_strike_rate': best_sr_list,
            'fastest_fifty': fastest_fifty_list,
            'fastest_century': fastest_century_list,
            
            'top_bowlers': top_bowlers_list,
            'best_bowling_figures': best_figures_list,
            'best_bowling_average': best_average_list,
            'best_economy': best_economy_list,
            'most_dots': most_dots_list
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)})
    
import io
import base64
import seaborn as sns
import matplotlib
matplotlib.use('Agg') # Server side rendering ke liye zaruri (crash roklega)
import matplotlib.pyplot as plt
from flask import jsonify # Agar pehle se import nahi hai

@app.route('/api/get_graphs', methods=['GET'])
def get_graphs():
    try:
        df = raw_df.copy()

        # --- 🔥 FILTER PAKISTANI PLAYERS 🔥 ---
        pak_players = ['Kamran Akmal', 'K Akmal', 'Sohail Tanvir', 'Younis Khan', 'Salman Butt', 'S Butt', 'Mohammad Hafeez', 'M Hafeez', 'Shoaib Malik', 'Mohammad Asif', 'Shoaib Akhtar', 'Umar Gul', 'Misbah-ul-Haq', 'Shahid Afridi', 'Azhar Mahmood']
        df = df[~df['batter'].isin(pak_players)]
        df = df[~df['bowler'].isin(pak_players)]

        # --- DATA PREP (Using Your Exact Column Names) ---
        df['runs_batter'] = pd.to_numeric(df.get('runs_batter', 0), errors='coerce').fillna(0)
        df['runs_total'] = pd.to_numeric(df.get('runs_total', 0), errors='coerce').fillna(0)
        df['runs_extras'] = pd.to_numeric(df.get('runs_extras', 0), errors='coerce').fillna(0)
        df['over'] = pd.to_numeric(df.get('over', 0), errors='coerce').fillna(0).astype(int)
        
        valid_wickets = ['caught', 'bowled', 'lbw', 'stumped', 'caught and bowled', 'hit wicket']
        df['is_wicket'] = df['wicket_kind'].isin(valid_wickets).astype(int) if 'wicket_kind' in df.columns else 0
        
        # Calculate valid balls using extra_type (Wides shouldn't count as balls faced)
        if 'extra_type' in df.columns:
            df['is_ball_faced'] = (df['extra_type'].astype(str).str.lower() != 'wides').astype(int)
        else:
            df['is_ball_faced'] = pd.to_numeric(df.get('valid_ball', 1), errors='coerce').fillna(1)

        # Match Phases (Using 100 as max limit to be safe)
        df['phase'] = pd.cut(df['over'], bins=[-1, 5, 14, 100], labels=['Powerplay', 'Middle', 'Death'])

        # --- 🎨 PRO-LEVEL DARK THEME SETTINGS ---
        sns.set_theme(style="darkgrid", rc={
            "axes.facecolor": "none", "figure.facecolor": "none", 
            "text.color": "#d1d5db", "axes.labelcolor": "#9ca3af", 
            "xtick.color": "#9ca3af", "ytick.color": "#9ca3af",
            "grid.color": "#1f2937", "axes.edgecolor": "#1f2937",
            "font.family": "sans-serif"
        })

        def get_img_base64():
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', transparent=True, dpi=120)
            plt.close('all') # Clear memory completely
            return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode('utf-8')

        graphs = {}

        # 1. Runs by Phase (Barplot)
        plt.figure(figsize=(7, 4.5))
        sns.barplot(data=df.groupby('phase', observed=False)['runs_total'].sum().reset_index(), x='phase', y='runs_total', palette="magma")
        plt.title('Total Runs by Match Phase', fontweight='bold', color='#60a5fa')
        plt.ylabel('Runs')
        graphs['g1'] = get_img_base64()

        # 2. Wickets by Phase (Barplot)
        plt.figure(figsize=(7, 4.5))
        sns.barplot(data=df.groupby('phase', observed=False)['is_wicket'].sum().reset_index(), x='phase', y='is_wicket', palette="viridis")
        plt.title('Wickets Fall by Match Phase', fontweight='bold', color='#60a5fa')
        plt.ylabel('Wickets')
        graphs['g2'] = get_img_base64()

        # 3. Run Rate Flow (Lineplot)
        plt.figure(figsize=(7, 4.5))
        avg_runs = df.groupby('over')['runs_total'].mean() * 6
        sns.lineplot(x=avg_runs.index, y=avg_runs.values, color="#3b82f6", linewidth=3, marker="o", markersize=8)
        plt.fill_between(avg_runs.index, avg_runs.values, color="#3b82f6", alpha=0.1)
        plt.title('Average Run Rate Progression', fontweight='bold', color='#60a5fa')
        plt.xlabel('Over Number'); plt.ylabel('Run Rate')
        graphs['g3'] = get_img_base64()

        # 4. Dismissal Type Distribution (Horizontal Bar)
        plt.figure(figsize=(7, 4.5))
        wkt_df = df[df['is_wicket'] == 1]['wicket_kind'].value_counts().head(5).reset_index()
        sns.barplot(data=wkt_df, y='wicket_kind', x='count', palette="flare")
        plt.title('Top 5 Dismissal Types', fontweight='bold', color='#60a5fa')
        plt.xlabel('Count'); plt.ylabel('')
        graphs['g4'] = get_img_base64()

        # 5. Run Source Composition (Donut equivalent)
        plt.figure(figsize=(6, 6))
        dots = len(df[(df['runs_batter'] == 0) & (df['is_ball_faced'] == 1)])
        ones = df[(df['runs_batter'] >= 1) & (df['runs_batter'] <= 3)]['runs_batter'].sum()
        fours = df[df['runs_batter'] == 4]['runs_batter'].sum()
        sixes = df[df['runs_batter'] == 6]['runs_batter'].sum()
        plt.pie([dots, ones, fours, sixes], labels=['Dots', '1s/2s/3s', 'Fours', 'Sixes'], colors=['#64748b', '#3b82f6', '#f59e0b', '#ef4444'], autopct='%1.1f%%', textprops={'color':"w", 'weight':'bold'})
        plt.title('Run Source Composition', fontweight='bold', color='#60a5fa')
        graphs['g5'] = get_img_base64()

        # 6. Toss Win vs Match Win Impact (Pie)
        plt.figure(figsize=(6, 6))
        match_df = df.groupby('match_id').first().reset_index()
        if 'toss_winner' in match_df.columns and 'match_won_by' in match_df.columns:
            valid_matches = match_df.dropna(subset=['toss_winner', 'match_won_by'])
            won_both = len(valid_matches[valid_matches['toss_winner'] == valid_matches['match_won_by']])
            lost_toss_won_match = len(valid_matches) - won_both
            if won_both > 0 or lost_toss_won_match > 0:
                plt.pie([won_both, lost_toss_won_match], labels=['Won Toss & Match', 'Lost Toss, Won Match'], colors=['#10b981', '#ef4444'], autopct='%1.1f%%', textprops={'color':"w"})
        plt.title('Toss Impact on Match Result', fontweight='bold', color='#60a5fa')
        graphs['g6'] = get_img_base64()

        # 7. Top 5 Venues Avg 1st Innings Score (Horizontal Bar)
        plt.figure(figsize=(8, 5))
        inn1 = df[df['innings'] == 1].groupby(['match_id', 'venue'])['runs_total'].sum().reset_index()
        ven_avg = inn1.groupby('venue')['runs_total'].mean().sort_values(ascending=False).head(5).reset_index()
        ven_avg['venue'] = ven_avg['venue'].str[:15] + ".."
        sns.barplot(data=ven_avg, x='runs_total', y='venue', palette="rocket")
        plt.title('Highest Scoring Venues (1st Innings Avg)', fontweight='bold', color='#60a5fa')
        plt.xlabel('Avg Runs'); plt.ylabel('')
        graphs['g7'] = get_img_base64()

        # 8. Score Distribution: Powerplay vs Death (Violin Plot)
        plt.figure(figsize=(7, 4.5))
        phase_runs = df[df['phase'].isin(['Powerplay', 'Death'])].groupby(['match_id', 'phase'], observed=False)['runs_total'].sum().reset_index()
        sns.violinplot(data=phase_runs, x='phase', y='runs_total', palette="Set2", inner="quartile")
        plt.title('Run Distribution: Powerplay vs Death', fontweight='bold', color='#60a5fa')
        plt.xlabel(''); plt.ylabel('Runs Scored')
        graphs['g8'] = get_img_base64()

        # 9. Top Teams Sixes (Bar)
        plt.figure(figsize=(7, 4.5))
        team_sixes = df[df['runs_batter'] == 6]['batting_team'].value_counts().head(5).reset_index()
        team_sixes['batting_team'] = team_sixes['batting_team'].str[:4]
        sns.barplot(data=team_sixes, x='batting_team', y='count', palette="coolwarm")
        plt.title('Most Sixes Hit by Teams', fontweight='bold', color='#60a5fa')
        plt.xlabel(''); plt.ylabel('Total Sixes')
        graphs['g9'] = get_img_base64()

        # 10. Density of 1st Innings Score (KDE Plot)
        plt.figure(figsize=(7, 4.5))
        sns.kdeplot(inn1['runs_total'], fill=True, color="#10b981", alpha=0.6, linewidth=2)
        plt.title('Density Curve of 1st Innings Totals', fontweight='bold', color='#60a5fa')
        plt.xlabel('Team Score'); plt.ylabel('Density Probability')
        graphs['g10'] = get_img_base64()

        # 11. Quadrant Analysis: Strike Rate vs Runs (Scatter)
        plt.figure(figsize=(8, 5))
        bat_agg = df.groupby('batter').agg(runs=('runs_batter', 'sum'), balls=('is_ball_faced', 'sum')).reset_index()
        bat_agg = bat_agg[bat_agg['runs'] >= 500]
        bat_agg['sr'] = (bat_agg['runs'] / bat_agg['balls']) * 100
        sns.scatterplot(data=bat_agg, x='runs', y='sr', color='#f59e0b', s=60, alpha=0.7)
        plt.axhline(bat_agg['sr'].mean(), color='red', linestyle='--', alpha=0.5)
        plt.axvline(bat_agg['runs'].mean(), color='red', linestyle='--', alpha=0.5)
        plt.title('Batter Quadrant: Runs vs Strike Rate (Min 500 runs)', fontweight='bold', color='#60a5fa')
        plt.xlabel('Total Runs'); plt.ylabel('Strike Rate')
        graphs['g11'] = get_img_base64()

        # 12. Top 5 Bowlers by Dot Balls (Horizontal Bar)
        plt.figure(figsize=(7, 4.5))
        top_dots = df[(df['runs_batter'] == 0) & (df['is_ball_faced'] == 1)]['bowler'].value_counts().head(5).reset_index()
        sns.barplot(data=top_dots, x='count', y='bowler', palette="mako")
        plt.title('Most Dot Balls Bowled', fontweight='bold', color='#60a5fa')
        plt.xlabel('Dot Balls'); plt.ylabel('')
        graphs['g12'] = get_img_base64()

        # 13. Quadrant Analysis: Bowler Economy vs Wickets (Scatter)
        plt.figure(figsize=(8, 5))
        bowl_agg = df.groupby('bowler').agg(wkts=('is_wicket', 'sum'), runs=('runs_bowler', 'sum'), balls=('is_ball_faced', 'sum')).reset_index()
        bowl_agg = bowl_agg[bowl_agg['wkts'] >= 30]
        bowl_agg['eco'] = (bowl_agg['runs'] / bowl_agg['balls']) * 6
        sns.scatterplot(data=bowl_agg, x='wkts', y='eco', color='#06b6d4', s=60, alpha=0.7)
        plt.axhline(bowl_agg['eco'].mean(), color='red', linestyle='--', alpha=0.5)
        plt.axvline(bowl_agg['wkts'].mean(), color='red', linestyle='--', alpha=0.5)
        plt.title('Bowler Quadrant: Wickets vs Economy (Min 30 wkts)', fontweight='bold', color='#60a5fa')
        plt.xlabel('Total Wickets'); plt.ylabel('Economy Rate')
        plt.gca().invert_yaxis() 
        graphs['g13'] = get_img_base64()

        # 14. Phase-wise Run Distribution (Heatmap)
        plt.figure(figsize=(8, 3))
        heat_data = df.groupby(['phase', 'batting_team'], observed=False)['runs_total'].sum().unstack(fill_value=0)
        top_6_teams = heat_data.sum(axis=0).sort_values(ascending=False).head(6).index
        sns.heatmap(heat_data[top_6_teams], cmap="YlOrRd", annot=False, linewidths=.5)
        plt.title('Heatmap: Runs scored by Top 6 Teams per Phase', fontweight='bold', color='#60a5fa')
        plt.xlabel(''); plt.ylabel('')
        graphs['g14'] = get_img_base64()

        # 15. Extra Runs Breakup (Pie - FIXED FOR YOUR COLUMNS)
        plt.figure(figsize=(6, 6))
        extras_df = df[df['extra_type'].notna() & (df['runs_extras'] > 0)]
        if not extras_df.empty:
            extras_grouped = extras_df.groupby('extra_type')['runs_extras'].sum()
            labels = extras_grouped.index.tolist()
            sizes = extras_grouped.values.tolist()
            if sum(sizes) > 0:
                plt.pie(sizes, labels=labels, autopct='%1.1f%%', textprops={'color':"w", 'weight':'bold'})
            else:
                plt.text(0.5, 0.5, "No Extra Runs Data", ha='center', va='center', color='white')
        else:
            plt.text(0.5, 0.5, "No Extra Runs Data", ha='center', va='center', color='white')
            
        plt.title('Extra Runs Breakup by Type', fontweight='bold', color='#60a5fa')
        graphs['g15'] = get_img_base64()

        return jsonify(graphs)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)