import pandas as pd

class IPLAnalytics:
    def __init__(self, df):
        self.df = df

    def get_venue_report(self, venue_name):
        # Filtering data for the specific venue
        v_df = self.df[self.df['venue'] == venue_name]
        if v_df.empty: return {"error": "Venue not found"}

        # Calculate Average Score (Final score of each match at this venue)
        avg_score = v_df.groupby(['match_id', 'innings'])['runs_total'].sum().mean()
        
        # Win stats (Bat 1st vs Chasing)
        matches = v_df.drop_duplicates(subset=['match_id'])
        total_m = len(matches)
        
        # Industry logic: match_won_by compare with toss_decision
        def check_chase_win(row):
            if row['toss_decision'] == 'field' and row['toss_winner'] == row['match_won_by']:
                return 1
            return 0
        
        chase_wins = matches.apply(check_chase_win, axis=1).sum()
        
        return {
            "avg_score": int(avg_score * 20 / 6), # Rough T20 estimate
            "chase_win_percentage": round((chase_wins/total_m)*100, 2),
            "total_matches": total_m
        }

    def get_h2h_players(self, batter, bowler):
        matchup = self.df[(self.df['batter'] == batter) & (self.df['bowler'] == bowler)]
        runs = matchup['runs_batter'].sum()
        outs = matchup['wicket_kind'].notnull().sum()
        return {"runs": int(runs), "outs": int(outs), "balls": len(matchup)}