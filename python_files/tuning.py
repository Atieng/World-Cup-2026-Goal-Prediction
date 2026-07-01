# ============================================================
# FIFA WORLD CUP 2026 — HYPERPARAMETER TUNING
# ============================================================
# Tunes: RandomForest Regressor, RandomForest Classifier,
#        XGBoost Regressor, XGBoost Classifier
# Method: GridSearchCV with StratifiedKFold
# ============================================================

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import (
    GridSearchCV, StratifiedKFold, cross_val_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error, f1_score, classification_report
)
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

DATA_PATH = 'data/'

# ============================================================
# STEP 1: LOAD ALL DATASETS
# ============================================================
print("=" * 60)
print("STEP 1: LOADING DATASETS")
print("=" * 60)

import os
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
        print(f"  {key:25s} → {dfs[key].shape}")
    except Exception as e:
        print(f"  {key:25s} → {e}")

# ============================================================
# STEP 2: FEATURE ENGINEERING
# ============================================================
print("\n" + "=" * 60)
print("STEP 2: FEATURE ENGINEERING")
print("=" * 60)

ta = dfs['team_appearances'].copy()

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

hosts = dfs['host_countries'][['tournament_id', 'team_id']].copy()
hosts['is_host'] = 1
base_perf = base_perf.merge(hosts, on=['tournament_id', 'team_id'], how='left')
base_perf['is_host'] = base_perf['is_host'].fillna(0).astype(int)

tourney = dfs['tournaments'][['tournament_id', 'year', 'count_teams',
                               'round_of_16', 'quarter_finals', 'semi_finals']].copy()
base_perf = base_perf.merge(tourney, on='tournament_id', how='left')
base_perf['era'] = pd.cut(base_perf['year'],
                           bins=[1929, 1965, 1985, 2000, 2030],
                           labels=[1, 2, 3, 4]).astype(float)

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

awards = dfs['award_winners'].groupby(['tournament_id', 'team_id']).size().reset_index(name='awards_won')
base_perf = base_perf.merge(awards, on=['tournament_id', 'team_id'], how='left')
base_perf['awards_won'] = base_perf['awards_won'].fillna(0)

bk = dfs['bookings'].groupby(['tournament_id', 'team_id']).agg(
    yellow_cards = ('yellow_card', 'sum'),
    red_cards    = ('red_card', 'sum'),
    sending_offs = ('sending_off', 'sum'),
).reset_index()
bk['discipline_index'] = bk['yellow_cards'] + (bk['red_cards'] * 3) + (bk['sending_offs'] * 2)
base_perf = base_perf.merge(bk, on=['tournament_id', 'team_id'], how='left')
base_perf[['yellow_cards', 'red_cards', 'sending_offs', 'discipline_index']] = \
    base_perf[['yellow_cards', 'red_cards', 'sending_offs', 'discipline_index']].fillna(0)

pk = dfs['penalty_kicks'].groupby(['tournament_id', 'team_id']).agg(
    penalties_taken  = ('converted', 'count'),
    penalties_scored = ('converted', 'sum'),
).reset_index()
pk['penalty_conversion_rate'] = pk['penalties_scored'] / (pk['penalties_taken'] + 1e-5)
base_perf = base_perf.merge(pk, on=['tournament_id', 'team_id'], how='left')
base_perf[['penalties_taken', 'penalties_scored', 'penalty_conversion_rate']] = \
    base_perf[['penalties_taken', 'penalties_scored', 'penalty_conversion_rate']].fillna(0)

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
base_perf[[c for c in mf_sel.columns if c != 'team_id']] = \
    base_perf[[c for c in mf_sel.columns if c != 'team_id']].fillna(0)

standings = dfs['tournament_standings'][['tournament_id', 'team_id', 'position']].copy()
standings.columns = ['tournament_id', 'team_id', 'final_position']
base_perf = base_perf.merge(standings, on=['tournament_id', 'team_id'], how='left')
base_perf['final_position'] = base_perf['final_position'].fillna(99)

print(f"  ✅ Feature matrix: {base_perf.shape}")

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
train_df = base_perf.merge(
    train_targets, on=['tournament_id', 'team_id'], how='inner'
).dropna(subset=['stage_clean', 'total_goals'])

print(f"  Training rows: {train_df.shape[0]}")

# ============================================================
# STEP 4: FEATURE SETS & MATRICES
# ============================================================
print("\n" + "=" * 60)
print("STEP 4: FEATURE SETS")
print("=" * 60)

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

for f in set(stage_features + goals_features):
    if f not in train_df.columns:
        train_df[f] = 0

X_stage = train_df[stage_features].fillna(0).values
X_goals = train_df[goals_features].fillna(0).values
y_stage = train_df['stage_numeric'].values.astype(int)
y_goals = train_df['total_goals'].values

scaler_stage = StandardScaler()
scaler_goals = StandardScaler()
X_stage_scaled = scaler_stage.fit_transform(X_stage)
X_goals_scaled = scaler_goals.fit_transform(X_goals)

sw  = compute_sample_weight(class_weight='balanced', y=y_stage)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print(f"  X_stage : {X_stage.shape}")
print(f"  X_goals : {X_goals.shape}")

# ============================================================
# STEP 5: BASELINE SCORES (before tuning)
# ============================================================
print("\n" + "=" * 60)
print("STEP 5: BASELINE SCORES (Before Tuning)")
print("=" * 60)

# Baseline Random Forest
rf_g_base = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_s_base = RandomForestClassifier(n_estimators=100, class_weight='balanced',
                                    random_state=42, n_jobs=-1)

base_rmse_rf = -cross_val_score(rf_g_base, X_goals_scaled, y_goals,
                                 cv=5, scoring='neg_root_mean_squared_error').mean()
base_f1_rf   = cross_val_score(rf_s_base, X_stage_scaled, y_stage,
                                cv=skf, scoring='f1_weighted').mean()

# Baseline XGBoost
xgb_g_base = xgb.XGBRegressor(n_estimators=100, random_state=42,
                                objective='count:poisson', verbosity=0)
xgb_s_base = xgb.XGBClassifier(n_estimators=100, random_state=42,
                                 objective='multi:softprob', num_class=6, verbosity=0)

base_rmse_xgb = -cross_val_score(xgb_g_base, X_goals_scaled, y_goals,
                                  cv=5, scoring='neg_root_mean_squared_error').mean()
base_f1_xgb   = cross_val_score(xgb_s_base, X_stage_scaled, y_stage,
                                 cv=skf, scoring='f1_weighted').mean()

print(f"  Random Forest  → Goals RMSE: {base_rmse_rf:.4f} | Stage F1: {base_f1_rf:.4f}")
print(f"  XGBoost        → Goals RMSE: {base_rmse_xgb:.4f} | Stage F1: {base_f1_xgb:.4f}")

# ============================================================
# STEP 6: RANDOM FOREST REGRESSOR TUNING
# ============================================================
print("\n" + "=" * 60)
print("STEP 6: RANDOM FOREST REGRESSOR — GridSearchCV")
print("=" * 60)

rf_reg_params = {
    'n_estimators':      [100, 200, 300],
    'max_depth':         [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf':  [1, 2, 4],
    'max_features':      ['sqrt', 'log2'],
}

rf_reg_grid = GridSearchCV(
    RandomForestRegressor(random_state=42, n_jobs=-1),
    rf_reg_params,
    cv=5,
    scoring='neg_root_mean_squared_error',
    n_jobs=-1,
    verbose=1,
    refit=True
)

print("  Running GridSearchCV for Random Forest Regressor...")
print(f"  Total combinations: {3 * 4 * 3 * 3 * 2} × 5 folds = "
      f"{3 * 4 * 3 * 3 * 2 * 5} fits")

rf_reg_grid.fit(X_goals_scaled, y_goals)

print(f"\n   Best RMSE      : {-rf_reg_grid.best_score_:.4f}")
print(f"  Best Params     :")
for k, v in rf_reg_grid.best_params_.items():
    print(f"    {k:25s}: {v}")

# ============================================================
# STEP 7: RANDOM FOREST CLASSIFIER TUNING
# ============================================================
print("\n" + "=" * 60)
print("STEP 7: RANDOM FOREST CLASSIFIER — GridSearchCV")
print("=" * 60)

rf_clf_params = {
    'n_estimators':      [100, 200, 300],
    'max_depth':         [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf':  [1, 2, 4],
    'max_features':      ['sqrt', 'log2'],
}

rf_clf_grid = GridSearchCV(
    RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1),
    rf_clf_params,
    cv=skf,
    scoring='f1_weighted',
    n_jobs=-1,
    verbose=1,
    refit=True
)

print("  Running GridSearchCV for Random Forest Classifier...")
rf_clf_grid.fit(X_stage_scaled, y_stage)

print(f"\n   Best F1        : {rf_clf_grid.best_score_:.4f}")
print(f"  Best Params     :")
for k, v in rf_clf_grid.best_params_.items():
    print(f"    {k:25s}: {v}")

# ============================================================
# STEP 8: XGBOOST REGRESSOR TUNING
# ============================================================
print("\n" + "=" * 60)
print("STEP 8: XGBOOST REGRESSOR — GridSearchCV")
print("=" * 60)

xgb_reg_params = {
    'n_estimators':    [200, 400, 600],
    'learning_rate':   [0.01, 0.05, 0.1],
    'max_depth':       [3, 5, 7],
    'subsample':       [0.7, 0.8, 1.0],
    'colsample_bytree':[0.7, 0.8, 1.0],
    'reg_alpha':       [0, 0.1, 0.5],
    'reg_lambda':      [1, 1.5, 2],
}

xgb_reg_grid = GridSearchCV(
    xgb.XGBRegressor(
        objective='count:poisson',
        random_state=42,
        verbosity=0,
        n_jobs=-1
    ),
    xgb_reg_params,
    cv=5,
    scoring='neg_root_mean_squared_error',
    n_jobs=-1,
    verbose=1,
    refit=True
)

print("  Running GridSearchCV for XGBoost Regressor...")
print(f"  Total combinations: {3*3*3*3*3*3*3} × 5 folds = "
      f"{3*3*3*3*3*3*3*5} fits")

xgb_reg_grid.fit(X_goals_scaled, y_goals)

print(f"\n   Best RMSE      : {-xgb_reg_grid.best_score_:.4f}")
print(f"  Best Params     :")
for k, v in xgb_reg_grid.best_params_.items():
    print(f"    {k:25s}: {v}")

# ============================================================
# STEP 9: XGBOOST CLASSIFIER TUNING
# ============================================================
print("\n" + "=" * 60)
print("STEP 9: XGBOOST CLASSIFIER — GridSearchCV")
print("=" * 60)

xgb_clf_params = {
    'n_estimators':    [200, 400, 600],
    'learning_rate':   [0.01, 0.05, 0.1],
    'max_depth':       [3, 5, 7],
    'subsample':       [0.7, 0.8, 1.0],
    'colsample_bytree':[0.7, 0.8, 1.0],
    'reg_alpha':       [0, 0.1, 0.5],
    'reg_lambda':      [1, 1.5, 2],
}

xgb_clf_grid = GridSearchCV(
    xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=6,
        random_state=42,
        verbosity=0,
        n_jobs=-1
    ),
    xgb_clf_params,
    cv=skf,
    scoring='f1_weighted',
    n_jobs=-1,
    verbose=1,
    refit=True
)

print("  Running GridSearchCV for XGBoost Classifier...")
xgb_clf_grid.fit(X_stage_scaled, y_stage, **{'sample_weight': sw})

print(f"\n  Best F1        : {xgb_clf_grid.best_score_:.4f}")
print(f"  Best Params     :")
for k, v in xgb_clf_grid.best_params_.items():
    print(f"    {k:25s}: {v}")

# ============================================================
# STEP 10: TUNED MODEL SCORES
# ============================================================
print("\n" + "=" * 60)
print("STEP 10: TUNED MODEL SCORES")
print("=" * 60)

# Validate tuned models with CV
tuned_rmse_rf  = -cross_val_score(
    rf_reg_grid.best_estimator_, X_goals_scaled, y_goals,
    cv=5, scoring='neg_root_mean_squared_error'
).mean()
tuned_f1_rf    = cross_val_score(
    rf_clf_grid.best_estimator_, X_stage_scaled, y_stage,
    cv=skf, scoring='f1_weighted'
).mean()
tuned_rmse_xgb = -cross_val_score(
    xgb_reg_grid.best_estimator_, X_goals_scaled, y_goals,
    cv=5, scoring='neg_root_mean_squared_error'
).mean()
tuned_f1_xgb   = cross_val_score(
    xgb_clf_grid.best_estimator_, X_stage_scaled, y_stage,
    cv=skf, scoring='f1_weighted'
).mean()

print(f"\n  {'Model':<25} {'Baseline RMSE':>15} {'Tuned RMSE':>12} {'Improvement':>13}")
print(f"  {'-'*65}")
print(f"  {'Random Forest (Goals)':<25} {base_rmse_rf:>15.4f} "
      f"{tuned_rmse_rf:>12.4f} "
      f"{'↑ +' + str(round(base_rmse_rf - tuned_rmse_rf, 4)):>13}")
print(f"  {'XGBoost (Goals)':<25} {base_rmse_xgb:>15.4f} "
      f"{tuned_rmse_xgb:>12.4f} "
      f"{'↑ +' + str(round(base_rmse_xgb - tuned_rmse_xgb, 4)):>13}")

print(f"\n  {'Model':<25} {'Baseline F1':>15} {'Tuned F1':>10} {'Improvement':>13}")
print(f"  {'-'*65}")
print(f"  {'Random Forest (Stage)':<25} {base_f1_rf:>15.4f} "
      f"{tuned_f1_rf:>10.4f} "
      f"{'↑ +' + str(round(tuned_f1_rf - base_f1_rf, 4)):>13}")
print(f"  {'XGBoost (Stage)':<25} {base_f1_xgb:>15.4f} "
      f"{tuned_f1_xgb:>10.4f} "
      f"{'↑ +' + str(round(tuned_f1_xgb - base_f1_xgb, 4)):>13}")

# ============================================================
# STEP 11: PICK BEST MODELS
# ============================================================
print("\n" + "=" * 60)
print("STEP 11: BEST MODELS SELECTED")
print("=" * 60)

best_goals_rmse = min(tuned_rmse_rf, tuned_rmse_xgb)
best_stage_f1   = max(tuned_f1_rf, tuned_f1_xgb)

if tuned_rmse_rf <= tuned_rmse_xgb:
    best_goals_model = rf_reg_grid.best_estimator_
    best_goals_name  = 'Random Forest Regressor (Tuned)'
else:
    best_goals_model = xgb_reg_grid.best_estimator_
    best_goals_name  = 'XGBoost Regressor (Tuned)'

if tuned_f1_rf >= tuned_f1_xgb:
    best_stage_model = rf_clf_grid.best_estimator_
    best_stage_name  = 'Random Forest Classifier (Tuned)'
else:
    best_stage_model = xgb_clf_grid.best_estimator_
    best_stage_name  = 'XGBoost Classifier (Tuned)'

print(f"  Best Goals Model : {best_goals_name}")
print(f"  Best RMSE        : {best_goals_rmse:.4f}")
print(f"\n  Best Stage Model : {best_stage_name}")
print(f"  Best F1          : {best_stage_f1:.4f}")

# ============================================================
# STEP 12: BUILD TEST FEATURES
# ============================================================
print("\n" + "=" * 60)
print("STEP 12: BUILD TEST FEATURES")
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

num_cols   = base_perf.select_dtypes(include=[np.number]).columns.tolist()
hist_avg   = base_perf.groupby('team_id')[num_cols].mean().reset_index()
test_feats = test_raw[['ID', 'country', 'team_id']].merge(
    hist_avg, on='team_id', how='left'
)
test_feats = test_feats.reset_index(drop=True)
test_feats['count_teams'] = 48
test_feats['is_host'] = test_feats['country'].isin(
    ['United States', 'Canada', 'Mexico']
).astype(int)
test_conf = test_raw[['team_id']].merge(
    dfs['teams'][['team_id', 'confederation_name']], on='team_id', how='left'
)
test_feats['confederation_strength'] = (
    test_conf['confederation_name'].map(conf_strength).fillna(2).values
)

all_feats = list(set(stage_features + goals_features))
bottom_25 = train_df[all_feats].quantile(0.25)
for col in all_feats:
    default_val = float(bottom_25.get(col, 0))
    if col not in test_feats.columns:
        test_feats[col] = default_val
    else:
        test_feats[col] = test_feats[col].fillna(default_val)

X_test_stage = scaler_stage.transform(test_feats[stage_features].fillna(0).values)
X_test_goals = scaler_goals.transform(test_feats[goals_features].fillna(0).values)

print(f"   X_test_stage: {X_test_stage.shape}")
print(f"   X_test_goals: {X_test_goals.shape}")

# ============================================================
# STEP 13: FINAL PREDICTIONS + 2026 CONSTRAINTS
# ============================================================
print("\n" + "=" * 60)
print("STEP 13: FINAL PREDICTIONS")
print("=" * 60)

def enforce_2026_constraints(stage_probs):
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

goals_pred        = np.round(
    np.clip(best_goals_model.predict(X_test_goals), 0, None)
).astype(int)
stage_probs       = best_stage_model.predict_proba(X_test_stage)
constrained_stage = enforce_2026_constraints(stage_probs)

print(f"  Goals — mean: {goals_pred.mean():.2f}  max: {goals_pred.max()}")
print(f"\n  Stage distribution:")
print(pd.Series(constrained_stage).value_counts().to_string())

# ============================================================
# STEP 14: SAVE SUBMISSION
# ============================================================
print("\n" + "=" * 60)
print("STEP 14: SUBMISSION")
print("=" * 60)

submission = pd.DataFrame({
    'ID':          test_raw['ID'].values,
    'total_goals': goals_pred,
    'Target':      constrained_stage
})
submission.to_csv('data/submission_tuned.csv', index=False)
print("  Figure 10 saved: fig10_submission.png")
print(f"\n  Preview:")
print(submission.head(10).to_string(index=False))

# ============================================================
# STEP 15: VISUALIZATION
# ============================================================
print("\n" + "=" * 60)
print("STEP 15: VISUALIZATION")
print("=" * 60)

plt.rcParams.update({
    'figure.facecolor': '#0d1117', 'axes.facecolor': '#161b22',
    'axes.edgecolor': '#30363d', 'axes.labelcolor': '#e6edf3',
    'xtick.color': '#8b949e', 'ytick.color': '#8b949e',
    'text.color': '#e6edf3', 'grid.color': '#21262d',
    'font.size': 12,
})
COLORS = ['#58a6ff', '#3fb950', '#f78166', '#d2a8ff', '#ffa657', '#79c0ff']

fig, axes = plt.subplots(1, 2, figsize=(18, 7))
fig.suptitle('HYPERPARAMETER TUNING RESULTS', fontsize=16,
             fontweight='bold', color='#58a6ff')

# Plot 1 — RMSE before vs after tuning
ax = axes[0]
models   = ['RF\n(Baseline)', 'RF\n(Tuned)', 'XGB\n(Baseline)', 'XGB\n(Tuned)']
rmse_vals = [base_rmse_rf, tuned_rmse_rf, base_rmse_xgb, tuned_rmse_xgb]
bar_colors = ['#58a6ff', '#3fb950', '#f78166', '#ffa657']
bars = ax.bar(models, rmse_vals, color=bar_colors, edgecolor='none', width=0.5)
ax.set_title('Goals RMSE — Before vs After Tuning\n(Lower is Better)',
             fontweight='bold', pad=15)
ax.set_ylabel('RMSE')
ax.set_ylim(0, max(rmse_vals) * 1.25)
for bar, val in zip(bars, rmse_vals):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.02,
            f'{val:.4f}', ha='center', va='bottom',
            fontsize=11, fontweight='bold')

# Plot 2 — F1 before vs after tuning
ax = axes[1]
f1_vals = [base_f1_rf, tuned_f1_rf, base_f1_xgb, tuned_f1_xgb]
bars = ax.bar(models, f1_vals, color=bar_colors, edgecolor='none', width=0.5)
ax.set_title('Stage F1 — Before vs After Tuning\n(Higher is Better)',
             fontweight='bold', pad=15)
ax.set_ylabel('F1 Score')
ax.set_ylim(0, 1.15)
for bar, val in zip(bars, f1_vals):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.01,
            f'{val:.4f}', ha='center', va='bottom',
            fontsize=11, fontweight='bold')

plt.tight_layout(pad=3.0)
plt.savefig('fig9_tuning_results.png', dpi=150,
            bbox_inches='tight', facecolor='#0d1117')
plt.show()
print("  Figure 9 saved: fig9_tuning_results.png")

# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "=" * 60)
print(" TUNING COMPLETE!")
print("=" * 60)
print(f"""
  TUNING RESULTS SUMMARY
  ────────────────────────────────────────────────────
  Model                    Baseline    Tuned    Gain
  ────────────────────────────────────────────────────
  RF Goals RMSE            {base_rmse_rf:.4f}    {tuned_rmse_rf:.4f}   {base_rmse_rf - tuned_rmse_rf:+.4f}
  XGB Goals RMSE           {base_rmse_xgb:.4f}    {tuned_rmse_xgb:.4f}   {base_rmse_xgb - tuned_rmse_xgb:+.4f}
  RF Stage F1              {base_f1_rf:.4f}    {tuned_f1_rf:.4f}   {tuned_f1_rf - base_f1_rf:+.4f}
  XGB Stage F1             {base_f1_xgb:.4f}    {tuned_f1_xgb:.4f}   {tuned_f1_xgb - base_f1_xgb:+.4f}
  ────────────────────────────────────────────────────

  BEST MODELS SELECTED
  ────────────────────────────────────────────────────
  Goals : {best_goals_name}  (RMSE: {best_goals_rmse:.4f})
  Stage : {best_stage_name}  (F1: {best_stage_f1:.4f})

  OUTPUT
  ────────────────────────────────────────────────────
  Submission → data/submission_tuned.csv
  Figure     → fig9_tuning_results.png

  NEXT STEPS
  ────────────────────────────────────────────────────
  → Build LightGBM + CatBoost ensemble
  → Submit and check leaderboard score
""")

# Save best params for reference
best_params_summary = {
    'rf_regressor':   rf_reg_grid.best_params_,
    'rf_classifier':  rf_clf_grid.best_params_,
    'xgb_regressor':  xgb_reg_grid.best_params_,
    'xgb_classifier': xgb_clf_grid.best_params_,
}

pd.DataFrame(best_params_summary).to_csv(
    'data/best_params.csv', index=True
)
print("   Best params saved: data/best_params.csv")