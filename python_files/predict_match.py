# ============================================================
# ARGENTINA vs EGYPT — ROUND OF 16 PREDICTION
# (adapted from your Spain vs Portugal script)
# ============================================================
import numpy as np
import pandas as pd
import joblib
from scipy.stats import poisson
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

DATA_PATH   = 'data/'
MODELS_PATH = 'models/'

# Load models
goals_model    = joblib.load(f'{MODELS_PATH}best_goals_model.joblib')
scaler_goals   = joblib.load(f'{MODELS_PATH}scaler_goals.joblib')
goals_features = joblib.load(f'{MODELS_PATH}goals_features.joblib')

print(f"✅ Model loaded: {type(goals_model).__name__}")

# Load data
ta    = pd.read_csv(f'{DATA_PATH}team_appearances.csv')
mf    = pd.read_csv(f'{DATA_PATH}master_features.csv')
gs    = pd.read_csv(f'{DATA_PATH}group_standings.csv')
bk    = pd.read_csv(f'{DATA_PATH}bookings.csv')
pk    = pd.read_csv(f'{DATA_PATH}penalty_kicks.csv')
teams = pd.read_csv(f'{DATA_PATH}teams.csv')

# ── Fix 1: Find correct team IDs ─────────────────────────────
print("\nTeam ID lookup:")
print(teams[teams['team_name'].str.contains(
    'Argentina|Egypt', case=False, na=False
)][['team_id', 'team_name', 'confederation_name']])

conf_strength = {
    'Union of European Football Associations':                                      5,
    'South American Football Confederation':                                        5,
    'Confederation of North, Central American and Caribbean Association Football':  3,
    'Confederation of African Football':                                            3,
    'Asian Football Confederation':                                                 2,
    'Oceania Football Confederation':                                               1,
}

def get_team_profile(team_name):
    team_row = teams[teams['team_name'] == team_name]
    if team_row.empty:
        print(f"❌ Not found: {team_name}")
        return None, None

    team_id = team_row['team_id'].values[0]
    conf    = team_row['confederation_name'].values[0]
    print(f"✅ {team_name} → team_id: {team_id}")

    team_ta = ta[ta['team_id'] == team_id]
    matches = len(team_ta)

    if matches == 0:
        print(f"⚠️  No match data for {team_name}")
        return None, None

    wins   = team_ta['win'].sum()
    losses = team_ta['lose'].sum()
    draws  = team_ta['draw'].sum()

    win_rate      = wins / matches
    loss_rate     = losses / matches
    draw_rate     = draws / matches
    points        = wins * 3 + draws
    pts_per_match = points / matches
    ko_rate       = team_ta['knockout_stage'].sum() / matches
    home_rate     = team_ta['home_team'].sum() / matches
    et_games      = team_ta['extra_time'].sum() / matches
    pen_games     = team_ta['penalty_shootout'].sum() / matches

    team_gs      = gs[gs['team_id'] == team_id]
    grp_pos      = team_gs['position'].mean()      if len(team_gs) > 0 else 3.0
    grp_pts      = team_gs['points'].mean()        if len(team_gs) > 0 else 3.0
    grp_played   = team_gs['played'].mean()        if len(team_gs) > 0 else 3.0
    grp_wins_avg = team_gs['wins'].mean()          if len(team_gs) > 0 else 1.0
    grp_win_rate = grp_wins_avg / grp_played       if grp_played > 0 else 0.0
    grp_pts_pg   = grp_pts / grp_played            if grp_played > 0 else 0.0
    grp_advanced = team_gs['advanced'].mean()      if len(team_gs) > 0 else 0.5

    team_bk  = bk[bk['team_id'] == team_id]
    yellow   = team_bk['yellow_card'].sum() / matches if matches > 0 else 0
    disc_idx = (team_bk['yellow_card'].sum() +
                team_bk['red_card'].sum() * 3 +
                team_bk['sending_off'].sum() * 2) / matches

    team_pk  = pk[pk['team_id'] == team_id]
    pen_taken= len(team_pk) / matches if matches > 0 else 0
    pen_rate = team_pk['converted'].sum() / len(team_pk) if len(team_pk) > 0 else 0

    team_mf = mf[mf['team_id'] == team_id]
    if len(team_mf) == 0:
        print(f"⚠️  No master features for {team_name}")
        return None, None

    hist = team_mf.iloc[0]

    profile = {
        'matches_played':            1,
        'wins':                      win_rate,
        'losses':                    loss_rate,
        'draws':                     draw_rate,
        'win_rate':                  win_rate,
        'loss_rate':                 loss_rate,
        'draw_rate':                 draw_rate,
        'points':                    pts_per_match,
        'points_per_match':          pts_per_match,
        'extra_time_games':          et_games,
        'penalty_shootouts':         pen_games,
        'knockout_rate':             ko_rate,
        'home_rate':                 home_rate,
        'is_host':                   0,
        'count_teams':               48,
        'confederation_strength':    conf_strength.get(conf, 3),
        'era':                       4,
        'group_position':            grp_pos,
        'group_points':              grp_pts,
        'group_win_rate':            grp_win_rate,
        'group_pts_per_game':        grp_pts_pg,
        'group_advanced':            grp_advanced,
        'yellow_cards':              yellow,
        'discipline_index':          disc_idx,
        'penalties_taken':           pen_taken,
        'penalty_conversion_rate':   pen_rate,
        'awards_won':                0,
        'hist_goals_avg':            hist.get('goals_avg', 0),
        'hist_goals_std':            hist.get('goals_std', 0),
        'hist_goals_median':         hist.get('goals_median', 0),
        'hist_goals_against_avg':    hist.get('goals_against_avg', 0),
        'hist_goal_diff_avg':        hist.get('goal_diff_avg', 0),
        'hist_recent_goals_avg':     hist.get('recent_goals_avg', 0),
        'hist_win_rate':             hist.get('win_rate', 0),
        'hist_win_total':            hist.get('win_total', 0),
        'hist_matches_played_total': hist.get('matches_played_total', 0),
        'hist_knockout_percentage':  hist.get('knockout_percentage', 0),
        'hist_deep_run_percentage':  hist.get('deep_run_percentage', 0),
        'hist_final_appearances':    hist.get('final_appearances', 0),
        'hist_semi_appearances':     hist.get('semi_appearances', 0),
        'hist_penalty_experience':   hist.get('penalty_experience', 0),
    }
    return profile, hist

# Build profiles
print("\n" + "=" * 50)
argentina_profile, argentina_hist = get_team_profile('Argentina')
egypt_profile,     egypt_hist     = get_team_profile('Egypt')

def predict_tournament_goals(profile):
    features = [profile.get(f, 0) for f in goals_features]
    X        = np.array(features).reshape(1, -1)
    X_scaled = scaler_goals.transform(X)
    pred     = goals_model.predict(X_scaled)[0]
    return max(0, pred)

argentina_pred = predict_tournament_goals(argentina_profile)
egypt_pred     = predict_tournament_goals(egypt_profile)

argentina_per_match = argentina_pred / 6
egypt_per_match     = egypt_pred / 6

# ── Poisson match simulation ─────────────────────────────────
argentina_xg = float(argentina_hist['goals_avg']) / max(float(argentina_hist['matches_played_total']), 1)
egypt_xg     = float(egypt_hist['goals_avg'])     / max(float(egypt_hist['matches_played_total']), 1)

argentina_xg = max(min(argentina_xg * 10, 3.0), 0.5)
egypt_xg     = max(min(egypt_xg * 10, 3.0), 0.5)

# ── NEW: blend model xG with this tournament's actual form ────
# Argentina have scored 11 goals in 4 games so far (2.75/game),
# Egypt have scored 5 in 4 games (1.25/game) with a leaky defense
# (5.4 xG against despite only 4 conceded). Uncomment to blend:
#
# argentina_xg = 0.6 * argentina_xg + 0.4 * 2.75
# egypt_xg     = 0.6 * egypt_xg     + 0.4 * 1.25

np.random.seed(42)
N              = 10000
argentina_wins = egypt_wins = draws = 0
scorelines     = []

for _ in range(N):
    a = poisson.rvs(argentina_xg)
    e = poisson.rvs(egypt_xg)
    scorelines.append((a, e))
    if a > e:   argentina_wins += 1
    elif e > a: egypt_wins     += 1
    else:       draws          += 1

argentina_pct = argentina_wins / N * 100
egypt_pct     = egypt_wins     / N * 100
draw_pct      = draws          / N * 100
top_scores    = Counter(scorelines).most_common(5)
avg_a         = np.mean([s[0] for s in scorelines])
avg_e         = np.mean([s[1] for s in scorelines])

print(f"""
╔══════════════════════════════════════════════╗
║   🏟️  ARGENTINA vs EGYPT — ROUND OF 16        ║
╠══════════════════════════════════════════════╣
║                                              ║
║  MODEL — Tournament Goals Predicted:         ║
║    🇦🇷 Argentina : {argentina_pred:>5.1f} goals             ║
║    🇪🇬 Egypt     : {egypt_pred:>5.1f} goals             ║
║                                              ║
║  Per Match Estimate:                         ║
║    🇦🇷 Argentina : {argentina_per_match:>4.2f} goals/match        ║
║    🇪🇬 Egypt     : {egypt_per_match:>4.2f} goals/match        ║
║                                              ║
╠══════════════════════════════════════════════╣
║  🎲 POISSON SIMULATION (10,000 matches):     ║
║                                              ║
║    🇦🇷 Argentina wins : {argentina_pct:>5.1f}%               ║
║    🇪🇬 Egypt wins     : {egypt_pct:>5.1f}%               ║
║    🤝 Draw            : {draw_pct:>5.1f}%               ║
║                                              ║
║  Average Score : Argentina {avg_a:.1f} - {avg_e:.1f} Egypt  ║
║                                              ║
║  Most Likely Scorelines:                     ║""")

for score, count in top_scores:
    prob = count / N * 100
    print(f"║    Argentina {score[0]}-{score[1]} Egypt → {prob:>4.1f}%              ║")

winner   = "🇦🇷 ARGENTINA" if argentina_pct > egypt_pct else "🇪🇬 EGYPT"
win_prob = max(argentina_pct, egypt_pct)

print(f"""║                                              ║
╠══════════════════════════════════════════════╣
║  🏆 PREDICTED WINNER : {winner:<22}║
║  📊 WIN PROBABILITY  : {win_prob:>5.1f}%               ║
╚══════════════════════════════════════════════╝
""")

print("=" * 50)
print("HISTORICAL STATS COMPARISON")
print("=" * 50)
print(f"""
  {'Metric':<28} {'Argentina':>10} {'Egypt':>10}
  {'─'*50}
  {'Win Rate (WC)':<28} {float(argentina_hist['win_rate']):>10.3f} {float(egypt_hist['win_rate']):>10.3f}
  {'Avg Goals/Tournament':<28} {float(argentina_hist['goals_avg']):>10.2f} {float(egypt_hist['goals_avg']):>10.2f}
  {'Final Appearances':<28} {int(argentina_hist['final_appearances']):>10} {int(egypt_hist['final_appearances']):>10}
  {'Semi Appearances':<28} {int(argentina_hist['semi_appearances']):>10} {int(egypt_hist['semi_appearances']):>10}
  {'Deep Run %':<28} {float(argentina_hist['deep_run_percentage']):>10.1f} {float(egypt_hist['deep_run_percentage']):>10.1f}
  {'Recent Win Rate':<28} {float(argentina_hist['recent_win_rate']):>10.3f} {float(egypt_hist['recent_win_rate']):>10.3f}
  {'─'*50}
""")

print("⚠️  Based on historical World Cup data only.")
print("   Does not include current 2026 form or injuries.")
print("\n📌 External model consensus (Opta/Kalshi, as of kickoff):")
print("   Argentina ~69-72% | Draw ~19-20% | Egypt ~9-12%")