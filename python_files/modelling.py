import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor   
from sklearn.model_selection import StratifiedKFold, cross_val_score 
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import(
    mean_squared_error, f1_score,classification_report,confusion_matrix)
 
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
DATA_PATH = "data/"   


 

# ============================================================
# STEP 1: LOAD ALL RAW DATASETS
# ============================================================
print("=" * 60)
print("STEP 1: LOADING DATASETS")
print("=" * 60)

dfs = {}
files = {
    'team_appearances':     'team_appearances.csv',
    'host_countries':       'host_countries.csv',
    'award_winners':        'award_winners.csv',
    'bookings':             'bookings.csv',
    'penalty_kicks':        'penalty_kicks.csv',
    'tournament_standings': 'tournament_standings.csv',
    'tournaments':          'tournaments.csv',
    'teams':                'teams.csv',
    'group_standings':      'group_standings.csv',
    'goals':                'goals.csv',
    'master_features':      'master_features.csv',
    'train':                'Train.csv',
    'test':                 'Test.csv',
}

for key, fname in files.items():
    try:
        dfs[key] = pd.read_csv(DATA_PATH + fname)
        print(f"  ✅ {key:25s} → {dfs[key].shape}")
    except Exception as e:
        print(f"  ❌ {key:25s} → {e}")

# ============================================================
# STEP 2: FEATURE ENGINEERING
# ============================================================
print("\n" + "=" * 60)
print("STEP 2: FEATURE ENGINEERING")
print("=" * 60)

ta = dfs['team_appearances'].copy()

# Base performance per tournament
base_perf = ta.groupby(['tournament_id', 'team_id', 'team_name']).agg(
    matches_played    = ('match_id', 'count'),
    goals_scored      = ('goals_for', 'sum'),
    goals_conceded    = ('goals_against', 'sum'),
    wins              = ('win', 'sum'),
    losses            = ('lose', 'sum'),
    draws             = ('draw', 'sum'),
    extra_time_games  = ('extra_time', 'sum'),
    penalty_shootouts = ('penalty_shootout', 'sum'),
    home_games        = ('home_team', 'sum'),
    knockout_games    = ('knockout_stage', 'sum'),
    group_games       = ('group_stage', 'sum'),
).reset_index()

base_perf['goal_difference']    = base_perf['goals_scored'] - base_perf['goals_conceded']
base_perf['goals_per_match']    = base_perf['goals_scored'] / base_perf['matches_played']
base_perf['conceded_per_match'] = base_perf['goals_conceded'] / base_perf['matches_played']
base_perf['win_rate']           = base_perf['wins'] / base_perf['matches_played']
base_perf['loss_rate']          = base_perf['losses'] / base_perf['matches_played']
base_perf['draw_rate']          = base_perf['draws'] / base_perf['matches_played']
base_perf['points']             = base_perf['wins'] * 3 + base_perf['draws']
base_perf['points_per_match']   = base_perf['points'] / base_perf['matches_played']
base_perf['goal_ratio']         = (base_perf['goals_scored'] + 1) / (base_perf['goals_conceded'] + 1)
base_perf['knockout_rate']      = base_perf['knockout_games'] / base_perf['matches_played']

# Host flag
hosts = dfs['host_countries'][['tournament_id', 'team_id']].copy()
hosts['is_host'] = 1
base_perf = base_perf.merge(hosts, on=['tournament_id', 'team_id'], how='left')
base_perf['is_host'] = base_perf['is_host'].fillna(0).astype(int)

# Tournament metadata
tourney = dfs['tournaments'][['tournament_id', 'year', 'count_teams',
                               'round_of_16', 'quarter_finals', 'semi_finals']].copy()
base_perf = base_perf.merge(tourney, on='tournament_id', how='left')
base_perf['era'] = pd.cut(base_perf['year'],
                           bins=[1929, 1965, 1985, 2000, 2030],
                           labels=[1, 2, 3, 4]).astype(float)

# Confederation
conf_strength = {
    'Union of European Football Associations':                                      5,
    'South American Football Confederation':                                        5,
    'Confederation of North, Central American and Caribbean Association Football':  3,
    'Confederation of African Football':                                            3,
    'Asian Football Confederation':                                                 2,
    'Oceania Football Confederation':                                               1,
}
teams_info = dfs['teams'][['team_id', 'confederation_name', 'region_name']]
base_perf  = base_perf.merge(teams_info, on='team_id', how='left')
base_perf['confederation_strength'] = base_perf['confederation_name'].map(conf_strength).fillna(2)

# Group standings
gs = dfs['group_standings'][['tournament_id', 'team_id', 'position', 'played',
                              'wins', 'draws', 'losses', 'goals_for',
                              'goals_against', 'goal_difference', 'points', 'advanced']].copy()
gs.columns = ['tournament_id', 'team_id', 'group_position', 'group_played',
              'group_wins', 'group_draws', 'group_losses', 'group_goals_for',
              'group_goals_against', 'group_goal_diff', 'group_points', 'group_advanced']
base_perf = base_perf.merge(gs, on=['tournament_id', 'team_id'], how='left')
base_perf['group_win_rate']     = base_perf['group_wins'] / (base_perf['group_played'] + 1e-5)
base_perf['group_goals_ratio']  = (base_perf['group_goals_for'] + 1) / (base_perf['group_goals_against'] + 1)
base_perf['group_pts_per_game'] = base_perf['group_points'] / (base_perf['group_played'] + 1e-5)

# Awards
awards = dfs['award_winners'].groupby(['tournament_id', 'team_id']).size().reset_index(name='awards_won')
base_perf = base_perf.merge(awards, on=['tournament_id', 'team_id'], how='left')
base_perf['awards_won'] = base_perf['awards_won'].fillna(0)

# Bookings
bk = dfs['bookings'].groupby(['tournament_id', 'team_id']).agg(
    yellow_cards = ('yellow_card', 'sum'),
    red_cards    = ('red_card', 'sum'),
    sending_offs = ('sending_off', 'sum'),
).reset_index()
bk['discipline_index'] = bk['yellow_cards'] + (bk['red_cards'] * 3) + (bk['sending_offs'] * 2)
base_perf = base_perf.merge(bk, on=['tournament_id', 'team_id'], how='left')
base_perf[['yellow_cards', 'red_cards', 'sending_offs', 'discipline_index']] = \
    base_perf[['yellow_cards', 'red_cards', 'sending_offs', 'discipline_index']].fillna(0)

# Penalties
pk = dfs['penalty_kicks'].groupby(['tournament_id', 'team_id']).agg(
    penalties_taken  = ('converted', 'count'),
    penalties_scored = ('converted', 'sum'),
).reset_index()
pk['penalty_conversion_rate'] = pk['penalties_scored'] / (pk['penalties_taken'] + 1e-5)
base_perf = base_perf.merge(pk, on=['tournament_id', 'team_id'], how='left')
base_perf[['penalties_taken', 'penalties_scored', 'penalty_conversion_rate']] = \
    base_perf[['penalties_taken', 'penalties_scored', 'penalty_conversion_rate']].fillna(0)

# Goals detail
g = dfs['goals'].copy()
goals_detail = g.groupby(['tournament_id', 'team_id']).agg(
    first_half_goals  = ('match_period', lambda x: (x == 'first half').sum()),
    second_half_goals = ('match_period', lambda x: (x == 'second half').sum()),
    own_goals_scored  = ('own_goal', 'sum'),
    penalty_goals     = ('penalty', 'sum'),
    early_goals       = ('minute_regulation', lambda x: (x <= 15).sum()),
    late_goals        = ('minute_regulation', lambda x: (x >= 75).sum()),
).reset_index()
goals_detail['late_goal_ability'] = goals_detail['late_goals'] / (
    goals_detail['first_half_goals'] + goals_detail['second_half_goals'] + 1e-5)
base_perf = base_perf.merge(goals_detail, on=['tournament_id', 'team_id'], how='left')
goal_cols = [c for c in goals_detail.columns if c not in ['tournament_id', 'team_id']]
base_perf[goal_cols] = base_perf[goal_cols].fillna(0)

# Historical master features
mf = dfs['master_features'].copy()
hist_cols = ['team_id', 'win_rate', 'win_total', 'draw_rate', 'loss_rate',
             'matches_played_total', 'penalty_experience',
             'final_appearances', 'semi_appearances',
             'quarter_appearances', 'round16_appearances', 'group_exits',
             'recent_win_rate', 'recent_goals_avg', 'recent_goal_diff_avg',
             'recent_stage_trend', 'avg_stage_reached', 'max_stage_reached',
             'knockout_percentage', 'deep_run_percentage', 'final_percentage',
             'goals_avg', 'goals_std', 'goals_median', 'goals_against_avg', 'goal_diff_avg']
mf_sel = mf[hist_cols].copy()
mf_sel.columns = ['team_id'] + ['hist_' + c for c in hist_cols[1:]]
base_perf = base_perf.merge(mf_sel, on='team_id', how='left')
hist_cols_renamed = [c for c in mf_sel.columns if c != 'team_id']
base_perf[hist_cols_renamed] = base_perf[hist_cols_renamed].fillna(0)

# Tournament standings
standings = dfs['tournament_standings'][['tournament_id', 'team_id', 'position']].copy()
standings.columns = ['tournament_id', 'team_id', 'final_position']
base_perf = base_perf.merge(standings, on=['tournament_id', 'team_id'], how='left')
base_perf['final_position'] = base_perf['final_position'].fillna(99)

print(f"  ✅ Feature matrix built: {base_perf.shape}")

# ============================================================
# STEP 3: TARGET MAPPING
# ============================================================
print("\n" + "=" * 60)
print("STEP 3: TARGET MAPPING")
print("=" * 60)

stage_map = {
    'group stage': 'group', 'group': 'group',
    'second group stage': 'roundof16',
    'round of 16': 'roundof16', 'roundof16': 'roundof16',
    'quarter-finals': 'qf', 'qf': 'qf',
    'semi-finals': 'sf', 'sf': 'sf', 'third-place match': 'sf',
    'final round': 'runnerup', 'runnerup': 'runnerup',
    'final': 'champion', 'champion': 'champion',
}
stage_numeric = {
    'group': 0, 'roundof16': 1, 'qf': 2,
    'sf': 3, 'runnerup': 4, 'champion': 5
}
inv_stage_map = {v: k for k, v in stage_numeric.items()}

train_raw = dfs['train'].copy()
train_raw['stage_clean']   = (train_raw['stage_reached']
                               .astype(str).str.lower().str.strip()
                               .map(stage_map).fillna('group'))
train_raw['stage_numeric'] = train_raw['stage_clean'].map(stage_numeric)

train_targets = train_raw[['tournament_id', 'team_id',
                            'stage_clean', 'stage_numeric', 'total_goals']].copy()

# Combine features + targets
train_df = base_perf.merge(
    train_targets, on=['tournament_id', 'team_id'], how='inner'
).dropna(subset=['stage_clean', 'total_goals'])

print(f"   Training rows   : {train_df.shape[0]}")
print(f"  Stage distribution:")
print(train_df['stage_clean'].value_counts().to_string())

# ============================================================
# STEP 4: DEFINE FEATURE SETS
# ============================================================
print("\n" + "=" * 60)
print("STEP 4: FEATURE SETS")
print("=" * 60)

# Stage features 
stage_features = [
    'matches_played', 'wins', 'losses', 'draws',
    'win_rate', 'loss_rate', 'draw_rate',
    'goals_conceded', 'conceded_per_match',
    'points', 'points_per_match', 'goal_ratio',
    'extra_time_games', 'penalty_shootouts', 'knockout_rate',
    'is_host', 'count_teams', 'confederation_strength', 'era',
    'group_position', 'group_points', 'group_win_rate',
    'group_goals_ratio', 'group_pts_per_game', 'group_advanced',
    'awards_won', 'yellow_cards', 'red_cards', 'discipline_index',
    'penalties_taken', 'penalty_conversion_rate',
    'first_half_goals', 'second_half_goals',
    'late_goal_ability', 'late_goals', 'early_goals',
    'hist_win_rate', 'hist_win_total', 'hist_matches_played_total',
    'hist_penalty_experience', 'hist_final_appearances',
    'hist_semi_appearances', 'hist_quarter_appearances',
    'hist_round16_appearances', 'hist_group_exits',
    'hist_recent_win_rate', 'hist_recent_stage_trend',
    'hist_avg_stage_reached', 'hist_max_stage_reached',
    'hist_knockout_percentage', 'hist_deep_run_percentage',
    'hist_final_percentage', 'hist_goals_avg',
    'hist_goals_against_avg', 'hist_goal_diff_avg',
    'final_position',
]

# Goals features — NO current tournament goals (avoid leakage)
goals_features = [
    'matches_played', 'wins', 'losses', 'draws',
    'win_rate', 'loss_rate', 'draw_rate',
    'goals_conceded', 'conceded_per_match',
    'points', 'points_per_match',
    'extra_time_games', 'penalty_shootouts', 'knockout_rate',
    'is_host', 'count_teams', 'confederation_strength', 'era',
    'group_position', 'group_points', 'group_win_rate',
    'group_pts_per_game', 'group_advanced',
    'awards_won', 'yellow_cards', 'discipline_index',
    'penalties_taken', 'penalty_conversion_rate',
    'hist_win_rate', 'hist_win_total', 'hist_matches_played_total',
    'hist_penalty_experience', 'hist_final_appearances',
    'hist_semi_appearances', 'hist_quarter_appearances',
    'hist_recent_win_rate', 'hist_avg_stage_reached',
    'hist_knockout_percentage', 'hist_deep_run_percentage',
    'hist_goals_avg', 'hist_goals_against_avg',
    'hist_goal_diff_avg', 'hist_goals_std', 'hist_goals_median',
    'final_position',
]

# Add missing columns as 0
for f in set(stage_features + goals_features):
    if f not in train_df.columns:
        train_df[f] = 0

# Final numpy arrays
X_stage = train_df[stage_features].fillna(0).values
X_goals = train_df[goals_features].fillna(0).values
y_stage = train_df['stage_numeric'].values.astype(int)
y_goals = train_df['total_goals'].values

print(f"  X_stage : {X_stage.shape}")
print(f"  X_goals : {X_goals.shape}")
print(f"  y_stage dist: {dict(zip(*np.unique(y_stage, return_counts=True)))}")
print(f"  y_goals mean: {y_goals.mean():.2f}")

# ============================================================
# STEP 5: BUILD TEST FEATURES
# ============================================================
print("\n" + "=" * 60)
print("STEP 5: BUILD TEST FEATURES")
print("=" * 60)

test_raw = dfs['test'].copy()

name_fixes = {
    'Czechia': 'Czech Republic', 'Turkiye': 'Turkey',
    'Cabo Verde': 'Cape Verde', "Cote d'Ivoire": 'Ivory Coast',
    'DR Congo': 'Congo DR',
}
test_raw['country_lookup'] = test_raw['country'].replace(name_fixes)

country_to_id = dfs['teams'].set_index('team_name')['team_id'].to_dict()
country_to_id.update(dfs['train'].set_index('country')['team_id'].to_dict())
test_raw['team_id'] = test_raw['country_lookup'].map(country_to_id)

print(f"  Matched  : {test_raw['team_id'].notna().sum()}/48")
print(f"  Unmatched: {test_raw[test_raw['team_id'].isna()]['country'].tolist()}")

# Historical averages per team
num_cols  = base_perf.select_dtypes(include=[np.number]).columns.tolist()
hist_avg  = base_perf.groupby('team_id')[num_cols].mean().reset_index()
test_feats = test_raw[['ID', 'country', 'team_id']].merge(
    hist_avg, on='team_id', how='left'
)

# 2026 overrides
test_feats['count_teams'] = 48
test_feats['is_host']     = test_feats['country'].isin(
    ['United States', 'Canada', 'Mexico']
).astype(int)

test_conf = test_raw[['team_id']].merge(
    dfs['teams'][['team_id', 'confederation_name']], on='team_id', how='left'
)
test_feats['confederation_strength'] = (
    test_conf['confederation_name'].map(conf_strength).fillna(2).values
)

# Fix duplicate index issue before filling
test_feats = test_feats.reset_index(drop=True)

# Fill new teams with bottom 25%
all_model_features = list(set(stage_features + goals_features))
bottom_25 = train_df[all_model_features].quantile(0.25)

for col in all_model_features:
    default_val = float(bottom_25.get(col, 0))
    if col not in test_feats.columns:
        test_feats[col] = default_val
    else:
        test_feats[col] = test_feats[col].fillna(default_val)

X_test_stage = test_feats[stage_features].fillna(0).values
X_test_goals = test_feats[goals_features].fillna(0).values

print(f"  X_test_stage: {X_test_stage.shape}")
print(f"  X_test_goals: {X_test_goals.shape}")

# ============================================================
# STEP 6: SCALE FEATURES
# ============================================================
print("\n" + "=" * 60)
print("STEP 6: SCALING")
print("=" * 60)

scaler_stage = StandardScaler()
scaler_goals = StandardScaler()

X_stage_scaled      = scaler_stage.fit_transform(X_stage)
X_goals_scaled      = scaler_goals.fit_transform(X_goals)
X_test_stage_scaled = scaler_stage.transform(X_test_stage)
X_test_goals_scaled = scaler_goals.transform(X_test_goals)

print("  Features scaled")

# ============================================================
# STEP 7: BASELINE MODELS
# ============================================================
print("\n" + "=" * 60)
print("STEP 7: BASELINE MODELS")
print("=" * 60)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ──   Linear Regression ────────────────────────────
print("\n[7.1] Linear Regression — Goals")
lr = LinearRegression()
cv_lr = cross_val_score(lr, X_goals_scaled, y_goals,
                        cv=5, scoring='neg_root_mean_squared_error')
print(f"  RMSE: {-cv_lr.mean():.4f} ± {cv_lr.std():.4f}")

# ──  Logistic Regression ──────────────────────────
print("\n[7.2] Logistic Regression — Stage")
log_reg = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
cv_log  = cross_val_score(log_reg, X_stage_scaled, y_stage,
                          cv=skf, scoring='f1_weighted')
print(f"  F1  : {cv_log.mean():.4f} ± {cv_log.std():.4f}")

# ──   Random Forest ─────────────────────────────────
print("\n[7.3] Random Forest — Goals")
rf_goals = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
cv_rf_g  = cross_val_score(rf_goals, X_goals_scaled, y_goals,
                            cv=5, scoring='neg_root_mean_squared_error')
print(f"  RMSE: {-cv_rf_g.mean():.4f} ± {cv_rf_g.std():.4f}")

print("\n[7.4] Random Forest — Stage")
rf_stage = RandomForestClassifier(n_estimators=300, class_weight='balanced',
                                   random_state=42, n_jobs=-1)
cv_rf_s  = cross_val_score(rf_stage, X_stage_scaled, y_stage,
                            cv=skf, scoring='f1_weighted')
print(f"  F1  : {cv_rf_s.mean():.4f} ± {cv_rf_s.std():.4f}")

# ── Upgraded 2: XGBoost ───────────────────────────────────────
print("\n[7.5] XGBoost — Goals")
xgb_goals_model = xgb.XGBRegressor(
    n_estimators=400, learning_rate=0.05, max_depth=5,
    subsample=0.8, colsample_bytree=0.8,
    objective='count:poisson',
    random_state=42, verbosity=0
)
cv_xgb_g = cross_val_score(xgb_goals_model, X_goals_scaled, y_goals,
                            cv=5, scoring='neg_root_mean_squared_error')
print(f"  RMSE: {-cv_xgb_g.mean():.4f} ± {cv_xgb_g.std():.4f}")

print("\n[7.6] XGBoost — Stage")
 
sw =compute_sample_weight(class_weight='balanced', y=y_stage)
xgb_stage_model = xgb.XGBClassifier(
    n_estimators=400, learning_rate=0.05, max_depth=5,
    subsample=0.8, colsample_bytree=0.8,
    objective='multi:softprob', num_class=6,
    random_state=42, verbosity=0
)
cv_xgb_s = cross_val_score(xgb_stage_model, X_stage_scaled, y_stage,
                            cv=skf, scoring='f1_weighted')
print(f"  F1  : {cv_xgb_s.mean():.4f} ± {cv_xgb_s.std():.4f}")

# ============================================================
# STEP 8: MODEL COMPARISON SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("STEP 8: MODEL COMPARISON")
print("=" * 60)

results = pd.DataFrame({
    'Model':  ['Linear Regression', 'Random Forest', 'XGBoost'],
    'RMSE':   [-cv_lr.mean(), -cv_rf_g.mean(), -cv_xgb_g.mean()],
    'F1':     [cv_log.mean(), cv_rf_s.mean(), cv_xgb_s.mean()],
})
results['Best_Goals'] = results['RMSE'] == results['RMSE'].min()
results['Best_Stage'] = results['F1']   == results['F1'].max()

print(results.to_string(index=False))

best_goals_model = results.loc[results['RMSE'].idxmin(), 'Model']
best_stage_model = results.loc[results['F1'].idxmax(), 'Model']
print(f"\n  Best Goals Model : {best_goals_model}")
print(f"  Best Stage Model : {best_stage_model}")

# ============================================================
# STEP 9: TRAIN FINAL BEST MODELS
# ============================================================
print("\n" + "=" * 60)
print("STEP 9: TRAIN FINAL MODELS")
print("=" * 60)

# Train best goals model
xgb_goals_model.fit(X_goals_scaled, y_goals)
print("  Goals model trained")

# Train best stage model
xgb_stage_model.fit(X_stage_scaled, y_stage, sample_weight=sw)
print("   Stage model trained")

# ============================================================
# STEP 10: PREDICTIONS + 2026 CONSTRAINTS
# ============================================================
print("\n" + "=" * 60)
print("STEP 10: PREDICTIONS")
print("=" * 60)

def enforce_2026_constraints(stage_probs):
    """Enforce exact 2026 tournament slot allocation"""
    constraints = {
        'champion': 1, 'runnerup': 1, 'sf': 2,
        'qf': 4, 'roundof16': 8, 'roundof32': 16, 'group': 16
    }
    col_map = {
        'group': 0, 'roundof16': 1, 'qf': 2,
        'sf': 3, 'runnerup': 4, 'champion': 5
    }
    order     = ['champion', 'runnerup', 'sf', 'qf', 'roundof16', 'roundof32', 'group']
    preds     = ['group'] * 48
    remaining = list(range(48))

    for stage in order:
        n = constraints[stage]
        if stage == 'roundof32':
            # Lowest roundof16 probability → roundof32
            scores = [(stage_probs[i, 1], i) for i in remaining]
            scores.sort(key=lambda x: x[0], reverse=False)
        else:
            c = col_map[stage]
            scores = [(stage_probs[i, c], i) for i in remaining]
            scores.sort(key=lambda x: x[0], reverse=True)

        chosen = [idx for _, idx in scores[:n]]
        for idx in chosen:
            preds[idx] = stage
            remaining.remove(idx)
    return preds

# Goals predictions
goals_pred = np.round(
    np.clip(xgb_goals_model.predict(X_test_goals_scaled), 0, None)
).astype(int)

# Stage predictions with constraints
stage_probs       = xgb_stage_model.predict_proba(X_test_stage_scaled)
constrained_stage = enforce_2026_constraints(stage_probs)

print(f"  Goals — mean: {goals_pred.mean():.2f}  max: {goals_pred.max()}")
print(f"\n  Stage distribution:")
stage_series = pd.Series(constrained_stage)
print(stage_series.value_counts().to_string())

# ============================================================
# STEP 11: BUILD & SAVE SUBMISSION
# ============================================================
print("\n" + "=" * 60)
print("STEP 11: SUBMISSION")
print("=" * 60)

submission = pd.DataFrame({
    'ID':          test_raw['ID'].values,
    'total_goals': goals_pred,
    'Target':      constrained_stage
})

submission.to_csv('data/submission_baseline.csv', index=False)
print("  Saved: data/submission_baseline.csv")
print(f"\n  Preview:")
print(submission.head(10).to_string(index=False))

# ============================================================
# STEP 12: VISUALIZE RESULTS
# ============================================================
print("\n" + "=" * 60)
print("STEP 12: VISUALIZATIONS")
print("=" * 60)

plt.rcParams.update({
    'figure.facecolor': '#0d1117', 'axes.facecolor': '#161b22',
    'axes.edgecolor': '#30363d', 'axes.labelcolor': '#e6edf3',
    'xtick.color': '#8b949e', 'ytick.color': '#8b949e',
    'text.color': '#e6edf3', 'grid.color': '#21262d',
})
COLORS = ['#58a6ff', '#3fb950', '#f78166', '#d2a8ff', '#ffa657', '#79c0ff']

 
# ── Figure 1: Model Comparison (top row) ─────────────────────
fig1, axes1 = plt.subplots(1, 3, figsize=(22, 7))
fig1.suptitle('BASELINE MODEL COMPARISON', fontsize=10,
              fontweight='bold', color='#58a6ff', y=1.02)

# Plot 1 — RMSE comparison
ax = axes1[0]
bars = ax.bar(results['Model'], results['RMSE'],
              color=['#58a6ff', '#3fb950', '#f78166'],
              edgecolor='none', width=0.5)
ax.set_title('Goals RMSE by Model\n(Lower is Better)',
             fontweight='bold', pad=15, fontsize=7  )
ax.set_ylabel('RMSE', labelpad=10)
ax.set_ylim(0, max(results['RMSE']) * 1.25)
ax.tick_params(axis='x', labelsize=12)
for bar, val in zip(bars, results['RMSE']):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.03,
            f'{val:.3f}', ha='center', va='bottom',
            fontsize=7, fontweight='bold',
            color='#e6edf3')

# Plot 2 — F1 comparison
ax = axes1[1]
bars = ax.bar(results['Model'], results['F1'],
              color=['#58a6ff', '#3fb950', '#f78166'],
              edgecolor='none', width=0.5)
ax.set_title('Stage F1 by Model\n(Higher is Better)',
             fontweight='bold', pad=15, fontsize=7  )
ax.set_ylabel('F1 Score', labelpad=10)
ax.set_ylim(0, 1.15)
ax.tick_params(axis='x', labelsize=12)
for bar, val in zip(bars, results['F1']):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.01,
            f'{val:.3f}', ha='center', va='bottom',
            fontsize=13, fontweight='bold',
            color='#e6edf3')

# Plot 3 — Stage distribution
ax = axes1[2]
stage_order  = ['group', 'roundof32', 'roundof16', 'qf', 'sf', 'runnerup', 'champion']
stage_counts = stage_series.value_counts().reindex(stage_order, fill_value=0)
bars = ax.bar(stage_counts.index, stage_counts.values,
              color=COLORS[:len(stage_counts)], edgecolor='none', width=0.6)
ax.set_title('2026 Predicted Stage Distribution',
             fontweight='bold', pad=15, fontsize=7)
ax.set_ylabel('Number of Teams', labelpad=10)
ax.tick_params(axis='x', rotation=35, labelsize=11)
for bar, val in zip(bars, stage_counts.values):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.15,
            str(val), ha='center', va='bottom',
            fontsize=12, fontweight='bold',
            color='#e6edf3')

fig1.tight_layout(pad=3.0)
plt.savefig('fig8a_model_comparison.png', dpi=150,
            bbox_inches='tight', facecolor='#0d1117')
plt.show()
print(" Figure 8a saved: fig8a_model_comparison.png")

# ── Figure 2: Feature Importances (bottom row) ───────────────
fig2, axes2 = plt.subplots(1, 3, figsize=(26, 9))
fig2.suptitle('FEATURE IMPORTANCE & GOAL PREDICTIONS',
              fontsize=10, fontweight='bold',
              color='#58a6ff', y=1.02)

# Plot 4 — Goals distribution
ax = axes2[0]
ax.hist(goals_pred, bins=15, color='#3fb950',
        edgecolor='#0d1117', alpha=0.85, rwidth=0.85)
ax.axvline(goals_pred.mean(), color='#f78166',
           linestyle='--', linewidth=2.5,
           label=f'Mean: {goals_pred.mean():.1f}')
ax.set_title('Predicted Goals Distribution (2026)',
             fontweight='bold', pad=15,fontsize=7)
ax.set_xlabel('Predicted Goals', labelpad=10)
ax.set_ylabel('Number of Teams', labelpad=10)
ax.legend(fontsize=12)

# Plot 5 — Goals feature importance
ax = axes2[1]
feat_imp = pd.DataFrame({
    'feature':    goals_features,
    'importance': xgb_goals_model.feature_importances_
}).sort_values('importance', ascending=False).head(15)
ax.barh(feat_imp['feature'].iloc[::-1],
        feat_imp['importance'].iloc[::-1],
        color='#ffa657', edgecolor='none', height=0.7)
ax.set_title('Top 15 Features — Goals Model',
             fontweight='bold', pad=15,fontsize=7)
ax.set_xlabel('Feature Importance', labelpad=10)
ax.tick_params(axis='y', labelsize=11)

# Plot 6 — Stage feature importance
ax = axes2[2]
feat_imp_s = pd.DataFrame({
    'feature':    stage_features,
    'importance': xgb_stage_model.feature_importances_
}).sort_values('importance', ascending=False).head(15)
ax.barh(feat_imp_s['feature'].iloc[::-1],
        feat_imp_s['importance'].iloc[::-1],
        color='#d2a8ff', edgecolor='none', height=0.7)
ax.set_title('Top 15 Features — Stage Model',
             fontweight='bold', pad=15, fontsize=7)
ax.set_xlabel('Feature Importance', labelpad=10)
ax.tick_params(axis='y', labelsize=11)

fig2.tight_layout(pad=3.0)
plt.savefig('fig8b_feature_importance.png', dpi=150,
            bbox_inches='tight', facecolor='#0d1117')
plt.show()
print(" Figure 8b saved: fig8b_feature_importance.png")

print("\n" + "=" * 60)
print(" MODELLING COMPLETE!")
print("=" * 60)
print(f"""
  RESULTS SUMMARY
  ─────────────────────────────────────────
  Model              Goals RMSE   Stage F1
  Linear Regression  {-cv_lr.mean():.4f}       {cv_log.mean():.4f}
  Random Forest      {-cv_rf_g.mean():.4f}       {cv_rf_s.mean():.4f}
  XGBoost            {-cv_xgb_g.mean():.4f}       {cv_xgb_s.mean():.4f}
  ─────────────────────────────────────────
  Best Goals : {best_goals_model}
  Best Stage : {best_stage_model}

  OUTPUT
  ─────────────────────────────────────────
  Submission → data/submission_baseline.csv
  Figure     → fig8_model_results.png
 
""")