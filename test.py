import pandas as pd
import numpy as np
import streamlit as st
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import math

# --- 1. FULL SQUAD DATA ---
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

TEAM_ALIASES = {
    'CSK': ['Chennai Super Kings'],
    'MI': ['Mumbai Indians'],
    'RCB': ['Royal Challengers Bangalore', 'Royal Challengers Bengaluru'],
    'KKR': ['Kolkata Knight Riders'],
    'DC': ['Delhi Capitals', 'Delhi Daredevils'],
    'GT': ['Gujarat Titans'],
    'LSG': ['Lucknow Super Giants'],
    'RR': ['Rajasthan Royals'],
    'SRH': ['Sunrisers Hyderabad', 'Deccan Chargers'],
    'PBKS': ['Punjab Kings', 'Kings XI Punjab']
}

PLAYER_NAME_MAP = {
    "KK Ahmed": "Khaleel Ahmed",
    "AK Hosein": "Akeal Hosein"
}

VENUE_CHASE_BIAS = {
    'M Chinnaswamy Stadium': 0.3,
    'Wankhede Stadium': 0.25,
    'Eden Gardens': 0.15,
    'MA Chidambaram Stadium': -0.3,
    'Ekana Cricket Stadium': -0.35,
    'Narendra Modi Stadium': 0.05,
    'Arun Jaitley Stadium': 0.1,
    'Sawai Mansingh Stadium': 0.0,
    'Rajiv Gandhi International Stadium': 0.0
}

# --- 2. PRO AI ENGINE ---
@st.cache_resource
def train_pro_ai_engine(file_path):
    try:
        raw_df = pd.read_csv(file_path)
        raw_df['runs_batter'] = raw_df['runs_batter'].fillna(0)
        raw_df['runs_bowler'] = raw_df['runs_bowler'].fillna(0)
        raw_df['bowler_wicket'] = raw_df['bowler_wicket'].fillna(0)
        
        if 'runs_total' not in raw_df.columns:
            if 'extras' in raw_df.columns:
                raw_df['runs_total'] = raw_df['runs_batter'] + raw_df['extras']
            else:
                raw_df['runs_total'] = raw_df['runs_batter']
                
        if 'date' in raw_df.columns:
            raw_df['date'] = pd.to_datetime(raw_df['date'], errors='coerce')
            raw_df['year'] = raw_df['date'].dt.year
            raw_df = raw_df.sort_values(by=['batter', 'date']) 

        current_year = 2026
        raw_df['weight'] = raw_df['year'].apply(lambda x: 1.0 if x >= (current_year - 2) else 0.4)
        stats = raw_df.groupby('batter')['runs_batter'].agg(['mean', 'std']).fillna(0)
        raw_df = raw_df.join(stats, on='batter', rsuffix='_stat')
        
        raw_df['norm_runs'] = np.where(raw_df['runs_batter'] > (raw_df['mean'] + 3 * raw_df['std']), 
                                       raw_df['mean'] + 2 * raw_df['std'], raw_df['runs_batter'])

        raw_df['recent_form'] = raw_df.groupby('batter')['norm_runs'].transform(lambda x: x.rolling(5, min_periods=1).mean())
        venue_dna = raw_df.groupby(['batter', 'venue'])['norm_runs'].mean().reset_index().rename(columns={'norm_runs': 'v_avg'})
        raw_df = raw_df.merge(venue_dna, on=['batter', 'venue'], how='left')
        matchup_dna = raw_df.groupby(['batter', 'bowling_team'])['norm_runs'].mean().reset_index().rename(columns={'norm_runs': 'o_avg'})
        raw_df = raw_df.merge(matchup_dna, on=['batter', 'bowling_team'], how='left')

        df_train = raw_df.groupby(['match_id', 'batter', 'batting_team', 'bowling_team', 'venue', 'innings']).agg({
            'norm_runs': 'sum', 'weight': 'first', 'recent_form': 'last', 'v_avg': 'last', 'o_avg': 'last'
        }).reset_index()

        le_team = LabelEncoder().fit(list(raw_df['batting_team'].unique()) + list(raw_df['bowling_team'].unique()) + ['Unknown'])
        le_venue = LabelEncoder().fit(list(raw_df['venue'].unique()) + ['Unknown'])
        le_player = LabelEncoder().fit(list(raw_df['batter'].unique()) + ['Unknown'])
        
        df_train['bat_team_id'] = le_team.transform(df_train['batting_team'])
        df_train['bowl_team_id'] = le_team.transform(df_train['bowling_team'])
        df_train['venue_id'] = le_venue.transform(df_train['venue'])
        df_train['player_id'] = le_player.transform(df_train['batter'])
        
        X = df_train[['bat_team_id', 'bowl_team_id', 'venue_id', 'player_id', 'recent_form', 'v_avg', 'o_avg', 'innings']].fillna(0)
        y = df_train['norm_runs']
        weights = df_train['weight']
        
        X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
            X, y, weights, test_size=0.2, random_state=42
        )
        
        model = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8, random_state=42).fit(X_train, y_train, sample_weight=w_train)
        
        inn1_scores = raw_df[raw_df['innings'] == 1].groupby(['match_id', 'venue', 'batting_team'])['runs_total'].sum().reset_index()
        venue_avg_scores = inn1_scores.groupby('venue')['runs_total'].mean().to_dict()
        team_avg_scores = inn1_scores.groupby('batting_team')['runs_total'].mean().to_dict()

        return raw_df, df_train, model, le_team, le_venue, le_player, venue_avg_scores, team_avg_scores
    except Exception as e:
        st.error(f"Engine Load Error: {e}")
        return None, None, None, None, None, None, None, None

def get_player_sr(player_name, raw_df):
    mapped_name = PLAYER_NAME_MAP.get(player_name, player_name)
    data = raw_df[raw_df['batter'] == mapped_name]
    if not data.empty and data['ball'].count() > 0:
        return (data['runs_batter'].sum() / data['ball'].count()) * 100
    return 110.0

def get_bowler_economy(player_name, raw_df):
    mapped_name = PLAYER_NAME_MAP.get(player_name, player_name)
    data = raw_df[raw_df['bowler'] == mapped_name]
    if not data.empty and data['ball'].count() > 0:
        return (data['runs_bowler'].sum() / (data['ball'].count() / 6))
    return 9.0 

# --- 3. UI DASHBOARD WITH NAVIGATION ---
st.set_page_config(page_title="IPL 2026 Pro Suite", layout="wide")
st.title("🏏 IPL 2026 - Deterministic AI Engine")

raw_df, train_df, model, le_team, le_venue, le_player, venue_avg, team_avg = train_pro_ai_engine('data/IPL.csv')

if raw_df is not None:
    st.sidebar.title("🎛️ Navigation")
    app_mode = st.sidebar.radio("Select Tool:", [
        "👤 Player Performance", 
        "📈 1st Inn Score Projector", 
        "🏆 Live Win Predictor", 
        "⚔️ Head-to-Head Battle",
        "🏢 Team vs Team Matchup"
    ])
    st.sidebar.markdown("---")

    # ==========================================
    # TOOL 1: PLAYER PERFORMANCE
    # ==========================================
    if app_mode == "👤 Player Performance":
        st.markdown("### 🔍 Player Career Statistics")
        col1, col2 = st.columns(2)
        with col1: sel_team = st.selectbox("Select Team", list(ipl_2026_squads.keys()))
        with col2: sel_player = st.selectbox("Select Player", ipl_2026_squads[sel_team])
        
        mapped_player = PLAYER_NAME_MAP.get(sel_player, sel_player)

        is_batter = mapped_player in raw_df['batter'].unique()
        is_bowler = mapped_player in raw_df['bowler'].unique()

        st.markdown("---")

        if is_batter or is_bowler:
            if is_batter:
                st.markdown(f"#### 🏏 {sel_player} - Batting Career")
                p_bat = raw_df[raw_df['batter'] == mapped_player]
                yearly = p_bat.groupby('year').agg({'match_id': 'nunique', 'runs_batter': 'sum', 'ball': 'count'}).rename(columns={'match_id': 'MAT', 'runs_batter': 'RUNS', 'ball': 'BF'})
                yearly['HS'] = p_bat.groupby(['year', 'match_id'])['runs_batter'].sum().groupby('year').max()
                yearly['AVG'] = (yearly['RUNS'] / yearly['MAT']).round(2)
                yearly['SR'] = ((yearly['RUNS'] / yearly['BF']) * 100).round(2)
                career = pd.DataFrame([{'MAT': yearly['MAT'].sum(), 'RUNS': yearly['RUNS'].sum(), 'HS': yearly['HS'].max(), 'AVG': (yearly['RUNS'].sum() / yearly['MAT'].sum()).round(2), 'BF': yearly['BF'].sum(), 'SR': ((yearly['RUNS'].sum() / yearly['BF'].sum()) * 100).round(2)}], index=['Career'])
                st.table(pd.concat([career, yearly.sort_index(ascending=False)]))

            if is_bowler:
                st.markdown(f"#### ☄️ {sel_player} - Bowling Career")
                p_bowl = raw_df[raw_df['bowler'] == mapped_player]
                yearly_b = p_bowl.groupby('year').agg({'match_id': 'nunique', 'ball': 'count', 'runs_bowler': 'sum', 'bowler_wicket': 'sum'}).rename(columns={'match_id': 'MAT', 'bowler_wicket': 'WKT'})
                yearly_b['Overs'] = yearly_b['ball'] // 6
                yearly_b['ECON'] = np.where(yearly_b['Overs'] > 0, (yearly_b['runs_bowler'] / yearly_b['Overs']).round(2), 0)
                
                total_balls = yearly_b['ball'].sum()
                total_overs = total_balls // 6
                career_b = pd.DataFrame([{'MAT': yearly_b['MAT'].sum(), 'Overs': total_overs, 'WKT': yearly_b['WKT'].sum(), 'ECON': round(yearly_b['runs_bowler'].sum() / total_overs, 2) if total_overs > 0 else 0}], index=['Career'])
                st.table(pd.concat([career_b, yearly_b[['MAT', 'Overs', 'WKT', 'ECON']].sort_index(ascending=False)]))
        else:
            st.info(f"🌟 **{sel_player}** is a **Potential Debutante**! No historical IPL data found.")

    # ==========================================
    # TOOL 2: LIVE SCORE PROJECTOR (UPGRADED TO DEEP DETERMINISTIC AI)
    # ==========================================
    elif app_mode == "📈 1st Inn Score Projector":
        st.markdown("### 📈 Deterministic 1st Innings Projector")
        st.caption("Analyzes the exact pitch state: Active batters, dugout depth, and remaining bowling threat.")
        
        c1, c2, c3 = st.columns(3)
        with c1: bat_team = st.selectbox("Batting Team", list(ipl_2026_squads.keys()), index=0)
        with c2: bowl_team = st.selectbox("Bowling Team", list(ipl_2026_squads.keys()), index=1)
        with c3: match_venue = st.selectbox("Venue", sorted(raw_df['venue'].unique()))

        st.markdown("#### 📋 Playing XI Setup")
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            playing_11_bat = st.multiselect(f"{bat_team} Initial XI", options=ipl_2026_squads[bat_team], default=ipl_2026_squads[bat_team][:11], max_selections=11)
        with col_t2:
            playing_11_bowl = st.multiselect(f"{bowl_team} Initial XI", options=ipl_2026_squads[bowl_team], default=ipl_2026_squads[bowl_team][:11], max_selections=11)

        if len(playing_11_bat) < 11 or len(playing_11_bowl) < 11:
            st.warning("⚠️ Please select 11 players for both teams.")
            st.stop()

        st.markdown("#### 🔄 Impact Player Updates")
        col_imp1, col_imp2 = st.columns(2)
        current_bat_xi = playing_11_bat.copy()
        with col_imp1:
            use_impact_bat = st.checkbox(f"Impact Sub used by {bat_team}?")
            if use_impact_bat:
                impact_in_bat = st.selectbox(f"Player IN ({bat_team})", [p for p in ipl_2026_squads[bat_team] if p not in playing_11_bat])
                impact_out_bat = st.selectbox(f"Player OUT ({bat_team})", playing_11_bat)
                if impact_in_bat == impact_out_bat: st.stop()
                if impact_out_bat in current_bat_xi: current_bat_xi.remove(impact_out_bat)
                current_bat_xi.append(impact_in_bat)

        current_bowl_xi = playing_11_bowl.copy()
        with col_imp2:
            use_impact_bowl = st.checkbox(f"Impact Sub used by {bowl_team}?")
            if use_impact_bowl:
                impact_in_bowl = st.selectbox(f"Player IN ({bowl_team})", [p for p in ipl_2026_squads[bowl_team] if p not in playing_11_bowl])
                impact_out_bowl = st.selectbox(f"Player OUT ({bowl_team})", playing_11_bowl)
                if impact_in_bowl == impact_out_bowl: st.stop()
                if impact_out_bowl in current_bowl_xi: current_bowl_xi.remove(impact_out_bowl)
                current_bowl_xi.append(impact_in_bowl)

        st.markdown("#### 🎯 Score & Overs")
        total_innings_overs = st.number_input("Total Overs in Innings (Adjust for Rain)", min_value=5, max_value=20, value=20)
        row1, row2, row3, row4 = st.columns(4)
        with row1: curr_score = st.number_input("Current Score", min_value=0, max_value=300, value=85)
        with row2: wkts = st.slider("Wickets Lost", min_value=0, max_value=10, value=2)
        with row3: o_comp = st.number_input("Overs Completed", min_value=0, max_value=total_innings_overs-1, value=10)
        with row4: b_comp = st.number_input("Balls (in current over)", min_value=0, max_value=5, value=0)

        balls_bowled = (o_comp * 6) + b_comp
        balls_left = (total_innings_overs * 6) - balls_bowled
        max_overs_remaining = math.ceil(balls_left / 6)
        safe_max = max(1, max_overs_remaining)

        st.markdown("#### 🏏 Crease Tracker (Who is batting?)")
        c_str, c_nstr, c_out = st.columns(3)
        with c_str: striker = st.selectbox("Striker", current_bat_xi, index=0)
        with c_nstr: 
            ns_options = [p for p in current_bat_xi if p != striker]
            non_striker = st.selectbox("Non-Striker", ns_options, index=0 if ns_options else None)
        
        out_options = [p for p in current_bat_xi if p not in [striker, non_striker]]
        with c_out:
            if wkts > 0:
                dismissed_batters = st.multiselect("Dugout (Dismissed)", out_options, max_selections=wkts)
            else:
                dismissed_batters = []
                st.info("No wickets lost yet.")

        st.markdown("#### ☄️ Bowling Threat (Who is bowling next?)")
        b3, b4 = st.columns(2)
        with b3: premium_death_overs = st.slider("Strike Bowlers Overs Left (Bumrah/Rashid type)?", min_value=0, max_value=safe_max, value=min(3, max_overs_remaining))
        with b4: weak_link_overs = st.slider("Part-timer Overs MUST bowl?", min_value=0, max_value=safe_max, value=0)

        if premium_death_overs + weak_link_overs > max_overs_remaining:
            st.error(f"⚠️ Logic Error: Allocated overs exceed remaining overs ({max_overs_remaining}).")
            st.stop()

        if st.button("Project Final Target"):
            if wkts >= 10 or balls_left <= 0:
                st.success(f"Innings Over! Final Score: {curr_score}")
            else:
                math_overs = balls_bowled / 6 if balls_bowled > 0 else 0.1667
                crr = curr_score / math_overs if balls_bowled > 0 else 0.0
                
                # 1. Calculate Venue & Team DNA Base Rate
                v_avg = venue_avg.get(match_venue, 165)
                bat_team_aliases = TEAM_ALIASES.get(bat_team, [bat_team])
                t_avg = next((team_avg[alias] for alias in bat_team_aliases if alias in team_avg), 165)
                base_rpb = ((v_avg + t_avg) / 2) / 120  # Base Runs Per Ball
                
                # 2. Calculate Active Batting Aggression
                s_sr = get_player_sr(striker, raw_df)
                ns_sr = get_player_sr(non_striker, raw_df)
                crease_aggression = ((s_sr + ns_sr) / 2) / 100.0  # e.g., 1.40 means 140 SR
                
                # 3. Calculate Dugout Depth (Who is left?)
                yet_to_bat = [p for p in current_bat_xi if p not in [striker, non_striker] + dismissed_batters]
                if yet_to_bat:
                    avg_depth_sr = sum([get_player_sr(p, raw_df) for p in yet_to_bat]) / len(yet_to_bat)
                else:
                    avg_depth_sr = 100.0
                depth_aggression = avg_depth_sr / 100.0
                
                # 4. Resources Left (Wickets)
                wkt_resource = math.pow((10 - wkts) / 10, 0.5) 
                
                # 5. Bowling Threat Logic
                overs_left = balls_left / 6
                if overs_left > 0:
                    death_density = min(premium_death_overs / overs_left, 1.0)
                    weak_density = min(weak_link_overs / overs_left, 1.0)
                else:
                    death_density = 0
                    weak_density = 0
                
                # High threat > 1 reduces runs. High weak density < 1 increases runs.
                bowling_threat = 1.0 + (death_density * 0.25) - (weak_density * 0.3)
                
                # 6. Phase Multiplier (Death overs are explosive)
                phase_mult = 1.0
                if overs_left <= 5:
                    phase_mult = 1.35 
                elif overs_left <= 10:
                    phase_mult = 1.15
                
                # --- THE DETERMINISTIC EQUATION ---
                expected_rpb = (base_rpb * ((crease_aggression * 0.5) + (depth_aggression * 0.5)) * wkt_resource * phase_mult) / bowling_threat
                
                projected_add = balls_left * expected_rpb
                final_score = int(curr_score + projected_add)
                projected_crr = final_score / total_innings_overs
                
                st.markdown("---")
                m1, m2, m3 = st.columns(3)
                m1.metric(label="Projected Final Score", value=f"{final_score} - {final_score + 10}")
                m2.metric(label="Current Run Rate", value=round(crr, 2))
                m3.metric(label="Projected Finish RR", value=round(projected_crr, 2))
                
                st.progress(min(final_score / 250, 1.0))
                
                st.info(f"🧠 **AI Insight:** With {wkts} down, the current crease SR is {int((s_sr+ns_sr)/2)} and dugout depth SR is {int(avg_depth_sr)}. Facing {int(death_density*100)}% strike bowlers in remaining overs.")

    # ==========================================
    # TOOL 3: LIVE WIN PREDICTOR 
    # ==========================================
    elif app_mode == "🏆 Live Win Predictor":
        st.markdown("### 🏆 Absolute Zero-Randomness Win Predictor")
        st.caption("Mathematically bounded. All Critical Flaws fixed & Clamped.")

        c1, c2, c3 = st.columns(3)
        with c1: chase_team = st.selectbox("Chasing Team", list(ipl_2026_squads.keys()), index=0)
        with c2: def_team = st.selectbox("Defending Team", list(ipl_2026_squads.keys()), index=1)
        with c3: venue = st.selectbox("Venue", sorted(raw_df['venue'].unique()))

        st.markdown("#### 📋 Playing XI & Context")
        col_t1, col_t2, col_t3 = st.columns(3)
        with col_t1:
            playing_11_chase = st.multiselect(f"{chase_team} Initial XI", options=ipl_2026_squads[chase_team], default=ipl_2026_squads[chase_team][:11], max_selections=11)
        with col_t2:
            playing_11_def = st.multiselect(f"{def_team} Initial XI", options=ipl_2026_squads[def_team], default=ipl_2026_squads[def_team][:11], max_selections=11)
        with col_t3:
            match_type = st.radio("Pressure Status", ["League Game", "Playoffs / Final"])

        if len(playing_11_chase) < 11 or len(playing_11_def) < 11:
            st.warning("⚠️ Please select 11 players for both teams.")
            st.stop()

        st.markdown("#### 🔄 Impact Player Rule (Live Swap)")
        col_imp1, col_imp2 = st.columns(2)
        current_chase_xi = playing_11_chase.copy()
        with col_imp1:
            use_impact_chase = st.checkbox(f"Activate Impact Sub for {chase_team}?")
            if use_impact_chase:
                impact_in_chase = st.selectbox(f"Player IN ({chase_team})", [p for p in ipl_2026_squads[chase_team] if p not in playing_11_chase])
                impact_out_chase = st.selectbox(f"Player OUT ({chase_team})", playing_11_chase)
                if impact_in_chase == impact_out_chase:
                    st.error("Impact IN and OUT cannot be the same player.")
                    st.stop()
                if impact_out_chase in current_chase_xi: current_chase_xi.remove(impact_out_chase)
                current_chase_xi.append(impact_in_chase)

        current_def_xi = playing_11_def.copy()
        with col_imp2:
            use_impact_def = st.checkbox(f"Activate Impact Sub for {def_team}?")
            if use_impact_def:
                impact_in_def = st.selectbox(f"Player IN ({def_team})", [p for p in ipl_2026_squads[def_team] if p not in playing_11_def])
                impact_out_def = st.selectbox(f"Player OUT ({def_team})", playing_11_def)
                if impact_in_def == impact_out_def:
                    st.error("Impact IN and OUT cannot be the same player.")
                    st.stop()
                if impact_out_def in current_def_xi: current_def_xi.remove(impact_out_def)
                current_def_xi.append(impact_in_def)

        st.markdown("#### 🎯 Core Match Situation")
        total_innings_overs = st.number_input("Total Overs in Innings (Adjust for Rain/DLS)", min_value=5, max_value=20, value=20)
        
        row1, row2, row3, row4 = st.columns(4)
        with row1: target = st.number_input("Target Score", min_value=50, max_value=300, value=185)
        with row2: score = st.number_input("Current Score", min_value=0, max_value=300, value=75)
        with row3: wkts = st.slider("Total Wickets Lost", min_value=0, max_value=10, value=2)
        with row4: 
            o1, o2 = st.columns(2)
            with o1: o_comp = st.number_input("Overs", min_value=0, max_value=total_innings_overs-1, value=10)
            with o2: b_comp = st.number_input("Balls", min_value=0, max_value=5, value=0)

        balls_bowled_so_far = (o_comp * 6) + b_comp
        total_balls_left = (total_innings_overs * 6) - balls_bowled_so_far
        max_overs_remaining = math.ceil(total_balls_left / 6)
        safe_max = max(1, max_overs_remaining) 

        st.markdown("#### 🌪️ Momentum & Constraints")
        d1, d2, d3 = st.columns(3)
        with d1: match_time = st.radio("Dew Factor", ["Night (Dew)", "Day (No Dew)"])
        with d2: recent_runs = st.number_input("Runs (Last 3 overs)", min_value=0, max_value=100, value=24)
        with d3: drought = st.number_input("Balls since last Boundary", min_value=0, max_value=max(1, int(balls_bowled_so_far)), value=0)

        st.markdown("#### 🏏 Crease Tracker")
        c_str, c_b1, c_nstr, c_lr = st.columns(4)
        with c_str: striker = st.selectbox("Striker", current_chase_xi, index=0)
        with c_b1: str_balls = st.number_input("Balls Faced by Striker", min_value=0, max_value=100, value=16) 
        with c_nstr: 
            ns_options = [p for p in current_chase_xi if p != striker]
            non_striker = st.selectbox("Non-Striker", ns_options, index=0 if ns_options else None)
        with c_lr: is_lr_combo = st.checkbox("Left-Right Hand Combo?", value=False)

        out_options = [p for p in current_chase_xi if p not in [striker, non_striker]]
        
        if wkts > 0:
            dismissed_batters = st.multiselect("Dugout (Dismissed Batters)", out_options, max_selections=wkts)
            if len(dismissed_batters) != wkts and wkts < 10:
                st.warning(f"⚠️ Warning: You selected {len(dismissed_batters)} out players, but 'Wickets Lost' is {wkts}.")
        else:
            dismissed_batters = []
            st.info("No wickets lost yet. Dugout selection disabled.")

        st.markdown("#### ☄️ Defending Bowling Allocation")
        b1, b2, b3, b4 = st.columns(4)
        with b1: current_bowler = st.selectbox("Current Bowler", current_def_xi, index=0)
        with b2: bowler_type = st.selectbox("Style", ["Pace", "Off-Spin", "Leg-Spin/Left-Arm Orthodox"])
        with b3: premium_death_overs = st.slider("Strike Bowler Overs Left?", min_value=0, max_value=safe_max, value=min(3, max_overs_remaining))
        with b4: weak_link_overs = st.slider("Part-timer Overs MUST bowl?", min_value=0, max_value=safe_max, value=0)

        if premium_death_overs + weak_link_overs > max_overs_remaining:
            st.error(f"⚠️ Logic Error: You allocated {premium_death_overs + weak_link_overs} overs, but only {max_overs_remaining} are left!")
            st.stop()

        if st.button("Calculate Exact Win Probability"):
            runs_needed = target - score
            
            if wkts >= 10 and runs_needed > 0:
                st.error(f"🎉 {def_team} wins! ({chase_team} is All Out)")
            elif runs_needed <= 0:
                st.success(f"🎉 {chase_team} wins!")
            elif total_balls_left <= 0:
                st.error(f"🎉 {def_team} wins!")
            else:
                math_overs = balls_bowled_so_far / 6 if balls_bowled_so_far > 0 else 0.1667
                rrr = min((runs_needed / total_balls_left) * 6, 36.0) if total_balls_left > 0 else 0.0
                crr = score / math_overs if balls_bowled_so_far > 0 else 0.0
                overs_left = total_balls_left / 6
                
                v_avg_score = venue_avg.get(venue, 165)
                v_par_rate = min(max(v_avg_score / 20, 7.5), 10.5) 
                
                venue_lower = venue.lower().replace(".", "")
                venue_bias_factor = next((val for key, val in VENUE_CHASE_BIAS.items() if key.lower().replace(".", "") in venue_lower), 0.0)
                
                recent_rr = (recent_runs / 3) if math_overs >= 3 else crr
                raw_momentum_gap = (rrr - recent_rr) / rrr if rrr > 0 else 0
                momentum_gap = max(min(raw_momentum_gap, 1.0), -1.0)
                
                recent_wkts = min(st.session_state.get('recent_wkts', 0), wkts) 
                collapse_penalty = recent_wkts * 0.45 
                
                phase_multiplier = 0.5 if math_overs < (total_innings_overs*0.3) else (1.0 if math_overs < (total_innings_overs*0.75) else 1.8)
                base_wicket_penalty = (math.pow(wkts, 1.2) * 0.15) * phase_multiplier

                matchup_modifier = 0.0
                if is_lr_combo and bowler_type == "Off-Spin": matchup_modifier = -0.3 
                elif not is_lr_combo and bowler_type == "Leg-Spin/Left-Arm Orthodox": matchup_modifier = 0.3 
                
                s_sr = get_player_sr(striker, raw_df)
                ns_sr = get_player_sr(non_striker, raw_df)
                
                max_sr = max(s_sr, ns_sr)
                set_bonus = (s_sr/100) * 0.4 if str_balls > 15 else (-0.2 if str_balls <= 5 else 0)
                
                raw_crease_bonus = ((max_sr - 135) * 0.015) + set_bonus
                crease_bonus = max(min(raw_crease_bonus, 1.5), -1.5)
                
                yet_to_bat = [p for p in current_chase_xi if p not in [striker, non_striker] + dismissed_batters]
                if yet_to_bat:
                    avg_depth_sr = sum([get_player_sr(p, raw_df) for p in yet_to_bat]) / len(yet_to_bat)
                else:
                    avg_depth_sr = 110.0
                
                raw_depth_bonus = (avg_depth_sr - 115) * 0.005 * (10 - wkts) 
                depth_bonus = max(min(raw_depth_bonus, 2.0), -2.0)
                
                cb_econ = get_bowler_economy(current_bowler, raw_df)
                cb_threat = (9.0 - cb_econ) * 0.1 
                
                if overs_left > 0:
                    death_bowling_density = min(premium_death_overs / overs_left, 1.0)
                    weak_link_density = min(weak_link_overs / overs_left, 1.0)
                else:
                    death_bowling_density = 0
                    weak_link_density = 0
                
                resource_threat = (death_bowling_density * 0.8) - (weak_link_density * 1.2)
                total_bowler_penalty = cb_threat + matchup_modifier + resource_threat

                base_pressure_gap = (rrr - v_par_rate) / v_par_rate 
                if base_pressure_gap > 0:
                    sustained_fatigue_multiplier = 1.0 + (overs_left / 10) 
                    pressure_gap = base_pressure_gap * sustained_fatigue_multiplier
                else:
                    pressure_gap = base_pressure_gap

                playoff_choke_penalty = 0.4 if ("Playoff" in match_type and target >= 170) else 0.0
                scoreboard_pressure = 0.5 if target >= 200 else 0.0
                dew_bonus = 0.3 if "Night" in match_time else 0.0
                
                raw_z = (pressure_gap * 2.5 
                     + base_wicket_penalty 
                     + collapse_penalty
                     + momentum_gap 
                     + scoreboard_pressure
                     + playoff_choke_penalty
                     + total_bowler_penalty 
                     - venue_bias_factor
                     - crease_bonus 
                     - depth_bonus 
                     - dew_bonus)
                
                z = max(min(raw_z, 5.0), -5.0)
                
                chase_prob = 1 / (1 + math.exp(z)) 
                chase_percent = int(chase_prob * 100)
                def_percent = 100 - chase_percent

                st.markdown("---")
                st.subheader("🧠 Absolute Deterministic Prediction")
                
                st.markdown(f"""
                <div style="width: 100%; background-color: #f87171; border-radius: 5px; display: flex; height: 35px; font-weight: bold; color: white; font-size: 18px;">
                    <div style="width: {chase_percent}%; background-color: #4ade80; display: flex; justify-content: center; align-items: center; border-radius: 5px 0 0 5px; transition: width 0.5s;">
                        {chase_team} ({chase_percent}%)
                    </div>
                    <div style="width: {def_percent}%; display: flex; justify-content: center; align-items: center; transition: width 0.5s;">
                        {def_team} ({def_percent}%)
                    </div>
                </div>
                <br>
                """, unsafe_allow_html=True)
                
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Runs Needed", f"{runs_needed} / {total_balls_left} b")
                c2.metric("Req Run Rate", round(rrr, 2), delta=round(crr - rrr, 2), delta_color="normal")
                c3.metric("True Depth SR", f"{int(avg_depth_sr)}")
                c4.metric("Z-Score Logic Bound", f"{round(z, 2)}")

    # ==========================================
    # TOOL 4: HEAD-TO-HEAD BATTLE
    # ==========================================
    elif app_mode == "⚔️ Head-to-Head Battle":
        st.markdown("### ⚔️ Player vs Team & Deep Matchups")
        st.caption("Deep dive into how a specific player performs against a franchise and its current 2026 squad.")

        c1, c2 = st.columns(2)
        with c1:
            team_a = st.selectbox("Select Player's Team", list(ipl_2026_squads.keys()), index=0)
            player_a = st.selectbox("Select Player", ipl_2026_squads[team_a])
        with c2:
            team_b = st.selectbox("Select Opponent Team", list(ipl_2026_squads.keys()), index=1)

        if team_a == team_b:
            st.warning("⚠️ Please select two different teams to view the battle.")
            st.stop()

        mapped_player = PLAYER_NAME_MAP.get(player_a, player_a)
        team_b_aliases = TEAM_ALIASES.get(team_b, [team_b])
        
        bat_data_vs_team = raw_df[(raw_df['batter'] == mapped_player) & (raw_df['bowling_team'].isin(team_b_aliases))]
        bowl_data_vs_team = raw_df[(raw_df['bowler'] == mapped_player) & (raw_df['batting_team'].isin(team_b_aliases))]

        bat_data_overall = raw_df[raw_df['batter'] == mapped_player]
        bowl_data_overall = raw_df[raw_df['bowler'] == mapped_player]

        st.markdown("---")

        if bat_data_vs_team.empty and bowl_data_vs_team.empty:
            st.info(f"🌟 No historical IPL face-offs found for **{player_a}** against **{team_b}** as a franchise.")
            st.stop()

        if not bat_data_vs_team.empty:
            st.markdown(f"#### 🏏 {player_a} Batting vs {team_b}")
            
            runs = bat_data_vs_team['runs_batter'].sum()
            balls = bat_data_vs_team['ball'].count()
            outs = bat_data_vs_team['bowler_wicket'].sum()
            inns = bat_data_vs_team['match_id'].nunique()
            hs = bat_data_vs_team.groupby('match_id')['runs_batter'].sum().max()
            
            avg = round(runs / outs, 2) if outs > 0 else (runs if runs > 0 else "N/A")
            sr = round((runs / balls) * 100, 2) if balls > 0 else 0
            
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Innings", inns)
            m2.metric("Runs", runs)
            m3.metric("Average", avg)
            m4.metric("Strike Rate", sr)
            m5.metric("Highest Score", hs)

            st.markdown(f"##### 🎯 vs Current {team_b} Bowlers (2026 Squad)")
            
            opp_squad_db_names = [PLAYER_NAME_MAP.get(p, p) for p in ipl_2026_squads[team_b]]
            
            matchup_data = []
            for display_name, db_name in zip(ipl_2026_squads[team_b], opp_squad_db_names):
                b_data = bat_data_overall[bat_data_overall['bowler'] == db_name] 
                
                if not b_data.empty:
                    r = b_data['runs_batter'].sum()
                    b = b_data['ball'].count()
                    w = b_data['bowler_wicket'].sum()
                    s_r = round((r/b)*100, 2) if b > 0 else 0.0
                    matchup_data.append({
                        'Opponent Bowler': display_name, 
                        'Runs Scored': r, 
                        'Balls Faced': b, 
                        'Dismissals': w, 
                        'Strike Rate': f"{s_r:.2f}" 
                    })
            
            if matchup_data:
                matchup_df = pd.DataFrame(matchup_data).sort_values(by='Runs Scored', ascending=False).reset_index(drop=True)
                st.table(matchup_df)
            else:
                st.info(f"No historical face-offs found between {player_a} and the current bowlers of {team_b}.")

        if not bowl_data_vs_team.empty:
            if not bat_data_vs_team.empty:
                st.markdown("---")
                
            st.markdown(f"#### ☄️ {player_a} Bowling vs {team_b}")
            
            runs_c = bowl_data_vs_team['runs_bowler'].sum()
            balls_b = bowl_data_vs_team['ball'].count()
            wkts = bowl_data_vs_team['bowler_wicket'].sum()
            inns_b = bowl_data_vs_team['match_id'].nunique()
            overs_b = balls_b // 6
            
            econ = round(runs_c / (balls_b / 6), 2) if balls_b > 0 else 0
            
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Innings", inns_b)
            m2.metric("Wickets", wkts)
            m3.metric("Economy", econ)
            m4.metric("Overs", overs_b)

            st.markdown(f"##### 🎯 vs Current {team_b} Batters (2026 Squad)")
            
            opp_squad_db_names = [PLAYER_NAME_MAP.get(p, p) for p in ipl_2026_squads[team_b]]
            
            matchup_data_b = []
            for display_name, db_name in zip(ipl_2026_squads[team_b], opp_squad_db_names):
                bat_data = bowl_data_overall[bowl_data_overall['batter'] == db_name] 
                
                if not bat_data.empty:
                    r = bat_data['runs_batter'].sum() 
                    b = bat_data['ball'].count()
                    w = bat_data['bowler_wicket'].sum()
                    s_r = round((r/b)*100, 2) if b > 0 else 0.0
                    matchup_data_b.append({
                        'Opponent Batter': display_name, 
                        'Runs Conceded': r, 
                        'Balls Bowled': b, 
                        'Wickets Taken': w, 
                        'Batter SR': f"{s_r:.2f}" 
                    })
            
            if matchup_data_b:
                matchup_df_b = pd.DataFrame(matchup_data_b).sort_values(by='Wickets Taken', ascending=False).reset_index(drop=True)
                st.table(matchup_df_b)
            else:
                st.info(f"No historical face-offs found between {player_a} and the current batters of {team_b}.")

    # ==========================================
    # TOOL 5: TEAM VS TEAM MATCHUP (NATIVE STREAMLIT UI)
    # ==========================================
    elif app_mode == "🏢 Team vs Team Matchup":
        st.markdown("### 🏢 Team vs Team: Head-to-Head & Form")
        st.caption("Analyze historical head-to-head records and recent tournament form using native components.")

        c1, c2 = st.columns(2)
        with c1:
            team_a = st.selectbox("Select Team 1", list(ipl_2026_squads.keys()), index=0)
        with c2:
            team_b = st.selectbox("Select Team 2", list(ipl_2026_squads.keys()), index=1)

        if team_a == team_b:
            st.warning("⚠️ Please select two different teams.")
            st.stop()

        team_a_aliases = TEAM_ALIASES.get(team_a, [team_a])
        team_b_aliases = TEAM_ALIASES.get(team_b, [team_b])

        if 'winning_team' in raw_df.columns:
            match_meta = raw_df.groupby('match_id').agg({
                'date': 'first',
                'batting_team': lambda x: list(x.unique()),
                'winning_team': 'first'
            }).reset_index()
            match_meta = match_meta[match_meta['batting_team'].apply(len) == 2]
            match_info = match_meta.rename(columns={'winning_team': 'winner'})
        else:
            if 'runs_total' not in raw_df.columns:
                raw_df['runs_total'] = raw_df['runs_batter'] + raw_df['extras'] if 'extras' in raw_df.columns else raw_df['runs_batter']
            
            match_summary = raw_df.groupby(['match_id', 'date', 'batting_team'])['runs_total'].sum().reset_index()
            idx_winner = match_summary.groupby('match_id')['runs_total'].idxmax()
            winners = match_summary.loc[idx_winner, ['match_id', 'date', 'batting_team']].rename(columns={'batting_team': 'winner'})
            
            match_teams = raw_df.groupby('match_id')['batting_team'].unique().reset_index()
            match_teams = match_teams[match_teams['batting_team'].apply(len) == 2]
            match_info = pd.merge(match_teams, winners[['match_id', 'date', 'winner']], on='match_id')

        match_info['date'] = pd.to_datetime(match_info['date'], errors='coerce')
        match_info = match_info.sort_values('date', ascending=False)
        
        def is_h2h(teams_array, aliases_a, aliases_b):
            has_a = any(alias in teams_array for alias in aliases_a)
            has_b = any(alias in teams_array for alias in aliases_b)
            return has_a and has_b

        h2h_matches = match_info[match_info['batting_team'].apply(lambda x: is_h2h(x, team_a_aliases, team_b_aliases))]
        
        total_matches = len(h2h_matches)
        wins_a = sum(h2h_matches['winner'].isin(team_a_aliases))
        wins_b = sum(h2h_matches['winner'].isin(team_b_aliases))
        ties = total_matches - (wins_a + wins_b)

        match_stats = []
        h2h_match_ids = h2h_matches['match_id'].tolist()

        for mid in h2h_match_ids:
            m_data = raw_df[raw_df['match_id'] == mid]
            year = m_data['year'].iloc[0]
            date = m_data['date'].iloc[0]
            
            inn1_data = m_data[m_data['innings'] == 1]
            inn2_data = m_data[m_data['innings'] == 2]
            
            t1 = inn1_data['batting_team'].iloc[0] if not inn1_data.empty else None
            t2 = inn2_data['batting_team'].iloc[0] if not inn2_data.empty else None
            
            if not t1 or not t2:
                continue 
                
            r1 = int(inn1_data['runs_total'].sum()) if not inn1_data.empty else 0
            w1 = int(inn1_data['bowler_wicket'].sum()) if not inn1_data.empty else 0
            
            r2 = int(inn2_data['runs_total'].sum()) if not inn2_data.empty else 0
            w2 = int(inn2_data['bowler_wicket'].sum()) if not inn2_data.empty else 0
            
            if r1 > r2:
                winner = t1
                margin = f"won by {r1 - r2} runs"
            elif r2 > r1:
                winner = t2
                wkts_left = 10 - w2
                margin = f"won by {wkts_left} wickets"
            else:
                winner = "Tie/No Result"
                margin = "Match Tied / No Result"
                
            match_stats.append({
                'match_id': mid,
                'date': pd.to_datetime(date),
                'year': year,
                'team1': t1, 'score1': r1,
                'team2': t2, 'score2': r2,
                'winner': winner,
                'margin_text': f"{winner} {margin}" if winner != "Tie/No Result" else margin
            })

        h2h_df = pd.DataFrame(match_stats)

        team_a_scores = []
        team_b_scores = []
        if not h2h_df.empty:
            for _, row in h2h_df.iterrows():
                if row['team1'] in team_a_aliases:
                    team_a_scores.append(row['score1'])
                    team_b_scores.append(row['score2'])
                else:
                    team_a_scores.append(row['score2'])
                    team_b_scores.append(row['score1'])

        highest_a = max(team_a_scores) if team_a_scores else 0
        lowest_a = min(team_a_scores) if team_a_scores else 0
        highest_b = max(team_b_scores) if team_b_scores else 0
        lowest_b = min(team_b_scores) if team_b_scores else 0

        st.markdown("---")
        st.markdown(f"#### ⚔️ Overall Head-to-Head: {team_a} vs {team_b}")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Matches Played", total_matches)
        col2.metric(f"{team_a} Won", wins_a)
        col3.metric(f"{team_b} Won", wins_b)
        col4.metric("No Result / Ties", ties)

        sc1, sc2, sc3, sc4 = st.columns(4)
        sc1.metric(f"{team_a} Highest Score", highest_a)
        sc2.metric(f"{team_a} Lowest Score", lowest_a)
        sc3.metric(f"{team_b} Highest Score", highest_b)
        sc4.metric(f"{team_b} Lowest Score", lowest_b)

        if total_matches > 0:
            last_match = h2h_matches.iloc[0]
            winner_alias = last_match['winner']
            if winner_alias in team_a_aliases:
                last_winner = team_a
            elif winner_alias in team_b_aliases:
                last_winner = team_b
            else:
                last_winner = "Tie/No Result"
            st.info(f"⏪ **Last Encounter:** **{last_winner}** won on {last_match['date'].strftime('%d %b %Y')}.")
        else:
            st.info(f"🌟 No historical matches found between {team_a} and {team_b}.")

        st.markdown("---")
        st.markdown("#### 🔥 Recent Form (Last 5 Matches)")
        
        def get_recent_form(team_aliases):
            team_matches = match_info[match_info['batting_team'].apply(lambda x: any(alias in x for alias in team_aliases))].head(5)
            form = []
            for _, row in team_matches.iterrows():
                if pd.isna(row['winner']):
                    form.append('NR')
                elif row['winner'] in team_aliases:
                    form.append('W')
                else:
                    form.append('L')
            return form

        form_a = get_recent_form(team_a_aliases)
        form_b = get_recent_form(team_b_aliases)

        def format_form(form_list):
            if not form_list:
                return "No recent matches."
            res = []
            for f in form_list:
                if f == 'W': res.append("🟢 W")
                elif f == 'L': res.append("🔴 L")
                else: res.append("⚪ NR")
            return " | ".join(res)

        f1, f2 = st.columns(2)
        f1.markdown(f"**{team_a}:** {format_form(form_a)}")
        f2.markdown(f"**{team_b}:** {format_form(form_b)}")

        st.markdown("---")
        st.markdown(f"#### 📅 {team_a} Vs {team_b} (Season by Season)")

        if not h2h_df.empty:
            seasons = h2h_df['year'].unique()
            for year in seasons:
                with st.expander(f"Season {year}", expanded=True):
                    season_matches = h2h_df[h2h_df['year'] == year]
                    for _, match in season_matches.iterrows():
                        margin_text = match["margin_text"]
                        icon = "⚪" if "Tie" in margin_text or "No Result" in margin_text else "🏏"
                        st.markdown(f"{icon} **{margin_text}**")