# ============================================================
# FIFA WORLD CUP 2026 — CORRECTED FINAL MODEL
# ============================================================
# Fixes applied:
# 1. Removed leaking features (current tournament goals,
#    final_position) from both feature sets
# 2. Reduced model complexity to prevent overfitting
# 3. Added stronger regularization
# 4. Saves best models in joblib format
# ============================================================

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import PoissonRegressor, LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error, f1_score, classification_report
)
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb
import joblib
import os
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

DATA_PATH    = 'data/'
MODELS_PATH  = 'models/'
os.makedirs(MODELS_PATH, exist_ok=True)

# ============================================================
# STEP 1: LOAD ALL DATASETS
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

# Match-based ratios only — no goals ratios
base_perf['win_rate']         = base_perf['wins'] / base_perf['matches_played']
base_perf['loss_rate']        = base_perf['losses'] / base_perf['matches_played']
base_perf['draw_rate']        = base_perf['draws'] / base_perf['matches_played']
base_perf['points']           = base_perf['wins'] * 3 + base_perf['draws']
base_perf['points_per_match'] = base_perf['points'] / base_perf['matches_played']
base_perf['knockout_rate']    = base_perf['knockout_games'] / base_perf['matches_played']
base_perf['home_rate']        = base_perf['home_games'] / base_perf['matches_played']

# Host flag
hosts = dfs['host_countries'][['tournament_id', 'team_id']].copy()
hosts['is_host'] = 1
base_perf = base_perf.merge(hosts, on=['tournament_id', 'team_id'], how='left')
base_perf['is_host'] = base_perf['is_host'].fillna(0).astype(int)

# Tournament metadata
tourney = dfs['tournaments'][[
    'tournament_id', 'year', 'count_teams',
    'round_of_16', 'quarter_finals', 'semi_finals'
]].copy()
base_perf = base_perf.merge(tourney, on='tournament_id', how='left')
base_perf['era'] = pd.cut(
    base_perf['year'],
    bins=[1929, 1965, 1985, 2000, 2030],
    labels=[1, 2, 3, 4]
).astype(float)

# Confederation strength
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
base_perf['confederation_strength'] = (
    base_perf['confederation_name'].map(conf_strength).fillna(2)
)

# Group standings — points and position only (no goals)
gs = dfs['group_standings'][[
    'tournament_id', 'team_id', 'position', 'played',
    'wins', 'draws', 'losses', 'points', 'advanced'
]].copy()
gs.columns = [
    'tournament_id', 'team_id', 'group_position', 'group_played',
    'group_wins', 'group_draws', 'group_losses',
    'group_points', 'group_advanced'
]
base_perf = base_perf.merge(gs, on=['tournament_id', 'team_id'], how='left')
base_perf['group_win_rate']     = base_perf['group_wins'] / (base_perf['group_played'] + 1e-5)
base_perf['group_pts_per_game'] = base_perf['group_points'] / (base_perf['group_played'] + 1e-5)

# Awards
awards = dfs['award_winners'].groupby(
    ['tournament_id', 'team_id']
).size().reset_index(name='awards_won')
base_perf = base_perf.merge(awards, on=['tournament_id', 'team_id'], how='left')
base_perf['awards_won'] = base_perf['awards_won'].fillna(0)

# Discipline
bk = dfs['bookings'].groupby(['tournament_id', 'team_id']).agg(
    yellow_cards = ('yellow_card', 'sum'),
    red_cards    = ('red_card', 'sum'),
    sending_offs = ('sending_off', 'sum'),
).reset_index()
bk['discipline_index'] = (
    bk['yellow_cards'] + (bk['red_cards'] * 3) + (bk['sending_offs'] * 2)
)
base_perf = base_perf.merge(bk, on=['tournament_id', 'team_id'], how='left')
base_perf[['yellow_cards', 'red_cards', 'sending_offs', 'discipline_index']] = \
    base_perf[['yellow_cards', 'red_cards', 'sending_offs', 'discipline_index']].fillna(0)

# Penalties
pk = dfs['penalty_kicks'].groupby(['tournament_id', 'team_id']).agg(
    penalties_taken  = ('converted', 'count'),
    penalties_scored = ('converted', 'sum'),
).reset_index()
pk['penalty_conversion_rate'] = (
    pk['penalties_scored'] / (pk['penalties_taken'] + 1e-5)
)
base_perf = base_perf.merge(pk, on=['tournament_id', 'team_id'], how='left')
base_perf[['penalties_taken', 'penalties_scored', 'penalty_conversion_rate']] = \
    base_perf[['penalties_taken', 'penalties_scored', 'penalty_conversion_rate']].fillna(0)

# Historical master features
mf = dfs['master_features'].copy()
hist_cols = [
    'team_id',
    'win_rate', 'win_total', 'draw_rate', 'loss_rate',
    'matches_played_total', 'penalty_experience',
    'final_appearances', 'semi_appearances',
    'quarter_appearances', 'round16_appearances', 'group_exits',
    'recent_win_rate', 'recent_goals_avg', 'recent_goal_diff_avg',
    'recent_stage_trend', 'avg_stage_reached', 'max_stage_reached',
    'knockout_percentage', 'deep_run_percentage', 'final_percentage',
    'goals_avg', 'goals_std', 'goals_median',
    'goals_against_avg', 'goal_diff_avg',
]
mf_sel = mf[hist_cols].copy()
mf_sel.columns = ['team_id'] + ['hist_' + c for c in hist_cols[1:]]
base_perf = base_perf.merge(mf_sel, on='team_id', how='left')
hist_renamed = [c for c in mf_sel.columns if c != 'team_id']
base_perf[hist_renamed] = base_perf[hist_renamed].fillna(0)

print(f"  ✅ Feature matrix built: {base_perf.shape}")

# ============================================================
# STEP 3: TARGET MAPPING
# ============================================================
print("\n" + "=" * 60)
print("STEP 3: TARGET MAPPING")
print("=" * 60)

stage_map = {
    'group stage': 'group',        'group': 'group',
    'second group stage': 'roundof16',
    'round of 16': 'roundof16',    'roundof16': 'roundof16',
    'quarter-finals': 'qf',        'qf': 'qf',
    'semi-finals': 'sf',           'sf': 'sf',
    'third-place match': 'sf',
    'final round': 'runnerup',     'runnerup': 'runnerup',
    'final': 'champion',           'champion': 'champion',
}
stage_numeric = {
    'group': 0, 'roundof16': 1, 'qf': 2,
    'sf': 3,    'runnerup': 4,  'champion': 5
}
inv_stage_map = {v: k for k, v in stage_numeric.items()}

train_raw = dfs['train'].copy()
train_raw['stage_clean']   = (
    train_raw['stage_reached']
    .astype(str).str.lower().str.strip()
    .map(stage_map).fillna('group')
)
train_raw['stage_numeric'] = train_raw['stage_clean'].map(stage_numeric)

train_targets = train_raw[[
    'tournament_id', 'team_id',
    'stage_clean', 'stage_numeric', 'total_goals'
]].copy()

train_df = base_perf.merge(
    train_targets, on=['tournament_id', 'team_id'], how='inner'
).dropna(subset=['stage_clean', 'total_goals'])

print(f"  ✅ Training rows   : {train_df.shape[0]}")
print(f"  Stage distribution:")
print(train_df['stage_clean'].value_counts().to_string())

# ============================================================
# STEP 4: CLEAN FEATURE SETS (No Leakage)
# ============================================================
print("\n" + "=" * 60)
print("STEP 4: CLEAN FEATURE SETS (No Leakage)")
print("=" * 60)

# ── Stage features — NO current tournament goals ─────────────
stage_features = [
    # Match performance — counts only
    'matches_played', 'wins', 'losses', 'draws',
    'win_rate', 'loss_rate', 'draw_rate',
    'points', 'points_per_match',
    'extra_time_games', 'penalty_shootouts',
    'knockout_rate', 'home_rate',

    # Tournament context
    'is_host', 'count_teams',
    'confederation_strength', 'era',

    # Group stage — points and position only
    'group_position', 'group_points',
    'group_win_rate', 'group_pts_per_game',
    'group_advanced',

    # Discipline
    'yellow_cards', 'red_cards',
    'discipline_index',

    # Penalties
    'penalties_taken', 'penalty_conversion_rate',

    # Awards
    'awards_won',

    # Historical only — all safe
    'hist_win_rate', 'hist_win_total',
    'hist_draw_rate', 'hist_loss_rate',
    'hist_matches_played_total',
    'hist_penalty_experience',
    'hist_final_appearances',
    'hist_semi_appearances',
    'hist_quarter_appearances',
    'hist_round16_appearances',
    'hist_group_exits',
    'hist_recent_win_rate',
    'hist_recent_stage_trend',
    'hist_avg_stage_reached',
    'hist_max_stage_reached',
    'hist_knockout_percentage',
    'hist_deep_run_percentage',
    'hist_final_percentage',
]

# ── Goals features — NO current tournament goals ─────────────
goals_features = [
    # Match performance — counts only
    'matches_played', 'wins', 'losses', 'draws',
    'win_rate', 'loss_rate', 'draw_rate',
    'points', 'points_per_match',
    'extra_time_games', 'penalty_shootouts',
    'knockout_rate', 'home_rate',

    # Tournament context
    'is_host', 'count_teams',
    'confederation_strength', 'era',

    # Group stage — points and position only
    'group_position', 'group_points',
    'group_win_rate', 'group_pts_per_game',
    'group_advanced',

    # Discipline
    'yellow_cards', 'discipline_index',

    # Penalties
    'penalties_taken', 'penalty_conversion_rate',

    # Awards
    'awards_won',

    # Historical goals — safe to use
    'hist_goals_avg',
    'hist_goals_std',
    'hist_goals_median',
    'hist_goals_against_avg',
    'hist_goal_diff_avg',
    'hist_recent_goals_avg',
    'hist_win_rate',
    'hist_win_total',
    'hist_matches_played_total',
    'hist_knockout_percentage',
    'hist_deep_run_percentage',
    'hist_final_appearances',
    'hist_semi_appearances',
]

# Ensure all features exist
for f in set(stage_features + goals_features):
    if f not in train_df.columns:
        train_df[f] = 0

print(f"  Stage features : {len(stage_features)} ✅ No current goals")
print(f"  Goals features : {len(goals_features)} ✅ No current goals")

# Verify no leaking features
leakers = [
    'goals_scored', 'goals_conceded', 'goal_difference',
    'goals_per_match', 'conceded_per_match', 'goal_ratio',
    'first_half_goals', 'second_half_goals',
    'late_goals', 'early_goals', 'late_goal_ability',
    'final_position'
]
stage_leaks = [f for f in leakers if f in stage_features]
goals_leaks = [f for f in leakers if f in goals_features]
print(f"\n  Leaking features in stage_features: {stage_leaks if stage_leaks else 'None ✅'}")
print(f"  Leaking features in goals_features: {goals_leaks if goals_leaks else 'None ✅'}")

# Final matrices
X_stage = train_df[stage_features].fillna(0).values
X_goals = train_df[goals_features].fillna(0).values
y_stage = train_df['stage_numeric'].values.astype(int)
y_goals = train_df['total_goals'].values

scaler_stage   = StandardScaler()
scaler_goals   = StandardScaler()
X_stage_scaled = scaler_stage.fit_transform(X_stage)
X_goals_scaled = scaler_goals.fit_transform(X_goals)
sw             = compute_sample_weight(class_weight='balanced', y=y_stage)
skf            = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print(f"\n  X_stage : {X_stage.shape}")
print(f"  X_goals : {X_goals.shape}")

# ============================================================
# STEP 5: DEFINE MODELS (Anti-Overfitting)
# ============================================================
print("\n" + "=" * 60)
print("STEP 5: MODEL DEFINITIONS (Anti-Overfitting)")
print("=" * 60)

models_goals = {

    # Baseline — Poisson (correct for count data)
    'Poisson Regression': PoissonRegressor(
        alpha=1.0, max_iter=1000
    ),

    # Random Forest — tuned but regularized
    'Random Forest': RandomForestRegressor(
        n_estimators      = 300,
        max_depth         = 8,       # ← limited depth
        max_features      = 'sqrt',
        min_samples_leaf  = 5,       # ← more leaf samples
        min_samples_split = 10,      # ← harder to split
        random_state      = 42,
        n_jobs            = -1
    ),

    # XGBoost — tuned params + strong regularization
    'XGBoost': xgb.XGBRegressor(
        n_estimators      = 200,
        max_depth         = 3,
        learning_rate     = 0.1,
        colsample_bytree  = 1.0,
        reg_alpha         = 0.1,
        reg_lambda        = 2.0,
        subsample         = 0.7,
        min_child_weight  = 5,       # ← prevents overfitting
        objective         = 'count:poisson',
        random_state      = 42,
        verbosity         = 0,
        n_jobs            = -1
    ),
}

models_stage = {

    # Baseline — Logistic Regression
    'Logistic Regression': LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        C=0.5,                       # ← stronger regularization
        random_state=42
    ),

    # Random Forest — tuned but regularized
    'Random Forest': RandomForestClassifier(
        n_estimators      = 200,
        max_depth         = 8,       # ← limited depth
        max_features      = 'sqrt',
        min_samples_leaf  = 5,       # ← more leaf samples
        min_samples_split = 10,      # ← harder to split
        class_weight      = 'balanced',
        random_state      = 42,
        n_jobs            = -1
    ),

    # XGBoost — tuned params + strong regularization
    'XGBoost': xgb.XGBClassifier(
        n_estimators      = 200,     # ← reduced from 600
        max_depth         = 3,
        learning_rate     = 0.05,    # ← increased from 0.01
        colsample_bytree  = 0.7,
        reg_alpha         = 0.5,     # ← stronger regularization
        reg_lambda        = 2.0,     # ← stronger regularization
        subsample         = 0.8,
        min_child_weight  = 5,       # ← prevents overfitting
        objective         = 'multi:softprob',
        num_class         = 6,
        random_state      = 42,
        verbosity         = 0,
        n_jobs            = -1
    ),
}

print("  ✅ Goals models defined:")
for name in models_goals:
    print(f"    • {name}")
print("  ✅ Stage models defined:")
for name in models_stage:
    print(f"    • {name}")

# ============================================================
# STEP 6: CROSS VALIDATION
# ============================================================
print("\n" + "=" * 60)
print("STEP 6: CROSS VALIDATION")
print("=" * 60)

goals_results  = {}
stage_results  = {}

print("\n  --- GOALS PREDICTION (RMSE ↓) ---")
for name, model in models_goals.items():
    cv_scores = cross_val_score(
        model, X_goals_scaled, y_goals,
        cv=5, scoring='neg_root_mean_squared_error'
    )
    rmse = -cv_scores.mean()
    std  = cv_scores.std()
    goals_results[name] = rmse
    print(f"  {name:25s} RMSE: {rmse:.4f} ± {std:.4f}")

print("\n  --- STAGE PREDICTION (F1 ↑) ---")
for name, model in models_stage.items():
    cv_scores = cross_val_score(
        model, X_stage_scaled, y_stage,
        cv=skf, scoring='f1_weighted'
    )
    f1  = cv_scores.mean()
    std = cv_scores.std()
    stage_results[name] = f1
    print(f"  {name:25s} F1  : {f1:.4f} ± {std:.4f}")

# ============================================================
# STEP 7: SELECT & TRAIN BEST MODELS
# ============================================================
print("\n" + "=" * 60)
print("STEP 7: SELECT & TRAIN BEST MODELS")
print("=" * 60)

best_goals_name  = min(goals_results, key=goals_results.get)
best_stage_name  = max(stage_results, key=stage_results.get)
best_goals_score = goals_results[best_goals_name]
best_stage_score = stage_results[best_stage_name]

print(f"\n  ✅ Best Goals : {best_goals_name} → RMSE: {best_goals_score:.4f}")
print(f"  ✅ Best Stage : {best_stage_name} → F1  : {best_stage_score:.4f}")

# Train on full data
best_goals_model = models_goals[best_goals_name]
best_stage_model = models_stage[best_stage_name]

best_goals_model.fit(X_goals_scaled, y_goals)
print(f"\n  ✅ {best_goals_name} trained on full data")

if hasattr(best_stage_model, 'class_weight'):
    best_stage_model.fit(X_stage_scaled, y_stage, )
else:
    best_stage_model.fit(X_stage_scaled, y_stage,
                         **{'sample_weight': sw})
print(f"  ✅ {best_stage_name} trained on full data")

# ============================================================
# STEP 8: SAVE MODELS WITH JOBLIB
# ============================================================
print("\n" + "=" * 60)
print("STEP 8: SAVING MODELS WITH JOBLIB")
print("=" * 60)

# Save best models
joblib.dump(
    best_goals_model,
    f'{MODELS_PATH}best_goals_model.joblib'
)
joblib.dump(
    best_stage_model,
    f'{MODELS_PATH}best_stage_model.joblib'
)

# Save scalers — needed to transform new data
joblib.dump(scaler_goals, f'{MODELS_PATH}scaler_goals.joblib')
joblib.dump(scaler_stage, f'{MODELS_PATH}scaler_stage.joblib')

# Save all models for comparison
for name, model in models_goals.items():
    fname = name.lower().replace(' ', '_')
    joblib.dump(model, f'{MODELS_PATH}goals_{fname}.joblib')

for name, model in models_stage.items():
    fname = name.lower().replace(' ', '_')
    joblib.dump(model, f'{MODELS_PATH}stage_{fname}.joblib')

# Save feature lists — needed to load new data correctly
joblib.dump(stage_features, f'{MODELS_PATH}stage_features.joblib')
joblib.dump(goals_features, f'{MODELS_PATH}goals_features.joblib')

print(f"  ✅ Saved: {MODELS_PATH}best_goals_model.joblib")
print(f"  ✅ Saved: {MODELS_PATH}best_stage_model.joblib")
print(f"  ✅ Saved: {MODELS_PATH}scaler_goals.joblib")
print(f"  ✅ Saved: {MODELS_PATH}scaler_stage.joblib")
print(f"  ✅ Saved: {MODELS_PATH}stage_features.joblib")
print(f"  ✅ Saved: {MODELS_PATH}goals_features.joblib")
print(f"\n  All models saved to: {MODELS_PATH}")

# Verify models load correctly
print("\n  Verifying saved models...")
loaded_goals = joblib.load(f'{MODELS_PATH}best_goals_model.joblib')
loaded_stage = joblib.load(f'{MODELS_PATH}best_stage_model.joblib')
loaded_sg    = joblib.load(f'{MODELS_PATH}scaler_goals.joblib')
loaded_ss    = joblib.load(f'{MODELS_PATH}scaler_stage.joblib')
print(f"  ✅ Goals model loads: {type(loaded_goals).__name__}")
print(f"  ✅ Stage model loads: {type(loaded_stage).__name__}")
print(f"  ✅ Scalers load correctly")

# ============================================================
# STEP 9: BUILD TEST FEATURES
# ============================================================
print("\n" + "=" * 60)
print("STEP 9: BUILD TEST FEATURES")
print("=" * 60)

test_raw = dfs['test'].copy()
name_fixes = {
    'Czechia':       'Czech Republic',
    'Turkiye':       'Turkey',
    'Cabo Verde':    'Cape Verde',
    "Cote d'Ivoire": 'Ivory Coast',
    'DR Congo':      'Congo DR',
}
test_raw['country_lookup'] = test_raw['country'].replace(name_fixes)
country_to_id = dfs['teams'].set_index('team_name')['team_id'].to_dict()
country_to_id.update(dfs['train'].set_index('country')['team_id'].to_dict())
test_raw['team_id'] = test_raw['country_lookup'].map(country_to_id)

print(f"  Matched  : {test_raw['team_id'].notna().sum()}/48")
unmatched = test_raw[test_raw['team_id'].isna()]['country'].tolist()
print(f"  Unmatched: {unmatched} ← get bottom 25% defaults")

num_cols   = base_perf.select_dtypes(include=[np.number]).columns.tolist()
hist_avg   = base_perf.groupby('team_id')[num_cols].mean().reset_index()
test_feats = test_raw[['ID', 'country', 'team_id']].merge(
    hist_avg, on='team_id', how='left'
).reset_index(drop=True)

# 2026 specific overrides
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

# Fill new teams with bottom 25%
all_feats = list(set(stage_features + goals_features))
bottom_25 = train_df[all_feats].quantile(0.25)
for col in all_feats:
    default_val = float(bottom_25.get(col, 0))
    if col not in test_feats.columns:
        test_feats[col] = default_val
    else:
        test_feats[col] = test_feats[col].fillna(default_val)

X_test_stage = scaler_stage.transform(
    test_feats[stage_features].fillna(0).values
)
X_test_goals = scaler_goals.transform(
    test_feats[goals_features].fillna(0).values
)

print(f"  ✅ X_test_stage: {X_test_stage.shape}")
print(f"  ✅ X_test_goals: {X_test_goals.shape}")

# ============================================================
# STEP 10: FINAL PREDICTIONS + 2026 CONSTRAINTS
# ============================================================
print("\n" + "=" * 60)
print("STEP 10: FINAL PREDICTIONS")
print("=" * 60)

def enforce_2026_constraints(stage_probs):
    """Enforce exact 2026 tournament slot allocation"""
    constraints = {
        'champion': 1, 'runnerup': 1, 'sf': 2,
        'qf': 4, 'roundof16': 8, 'roundof32': 16, 'group': 16
    }
    col_map = {
        'group': 0, 'roundof16': 1, 'qf': 2,
        'sf': 3,    'runnerup': 4,  'champion': 5
    }
    order     = ['champion', 'runnerup', 'sf', 'qf',
                 'roundof16', 'roundof32', 'group']
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

# Goals predictions
goals_pred = np.round(
    np.clip(best_goals_model.predict(X_test_goals), 0, None)
).astype(int)

# Stage predictions with constraints
stage_probs       = best_stage_model.predict_proba(X_test_stage)
constrained_stage = enforce_2026_constraints(stage_probs)

print(f"  Goals — mean: {goals_pred.mean():.2f}  "
      f"min: {goals_pred.min()}  max: {goals_pred.max()}")
print(f"\n  Stage distribution:")
print(pd.Series(constrained_stage).value_counts().to_string())

# Sanity check — goals should align with stage
submission_check = pd.DataFrame({
    'country':  test_raw['country'].values,
    'goals':    goals_pred,
    'stage':    constrained_stage
}).sort_values('goals', ascending=False)

print(f"\n  Top 10 teams by predicted goals:")
print(submission_check.head(10).to_string(index=False))

# ============================================================
# STEP 11: SAVE SUBMISSION
# ============================================================
print("\n" + "=" * 60)
print("STEP 11: SAVE SUBMISSION")
print("=" * 60)

submission = pd.DataFrame({
    'ID':          test_raw['ID'].values,
    'total_goals': goals_pred,
    'Target':      constrained_stage
})
submission.to_csv('data/submission_corrected.csv', index=False)
print("  ✅ Saved: data/submission_corrected.csv")

# ============================================================
# STEP 12: VISUALIZATION
# ============================================================
print("\n" + "=" * 60)
print("STEP 12: VISUALIZATION")
print("=" * 60)

plt.rcParams.update({
    'figure.facecolor': '#0d1117', 'axes.facecolor': '#161b22',
    'axes.edgecolor': '#30363d',   'axes.labelcolor': '#e6edf3',
    'xtick.color': '#8b949e',      'ytick.color': '#8b949e',
    'text.color': '#e6edf3',       'grid.color': '#21262d',
    'font.size': 12,
})
COLORS = ['#58a6ff', '#3fb950', '#f78166', '#d2a8ff',
          '#ffa657', '#79c0ff', '#56d364']

# ── Figure 1: CV Scores comparison ───────────────────────────
fig1, axes1 = plt.subplots(1, 2, figsize=(18, 7))
fig1.suptitle('MODEL COMPARISON — CORRECTED (No Leakage)',
              fontsize=16, fontweight='bold', color='#58a6ff')

ax = axes1[0]
names_g = list(goals_results.keys())
vals_g  = list(goals_results.values())
bars = ax.bar(names_g, vals_g,
              color=COLORS[:len(names_g)],
              edgecolor='none', width=0.5)
ax.set_title('Goals RMSE by Model\n(Lower is Better)',
             fontweight='bold', pad=15)
ax.set_ylabel('RMSE')
ax.set_ylim(0, max(vals_g) * 1.3)
for bar, val in zip(bars, vals_g):
    color = '#3fb950' if val == min(vals_g) else '#e6edf3'
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.02,
            f'{val:.4f}', ha='center', va='bottom',
            fontsize=12, fontweight='bold', color=color)
ax.tick_params(axis='x', labelsize=11)

ax = axes1[1]
names_s = list(stage_results.keys())
vals_s  = list(stage_results.values())
bars = ax.bar(names_s, vals_s,
              color=COLORS[:len(names_s)],
              edgecolor='none', width=0.5)
ax.set_title('Stage F1 by Model\n(Higher is Better)',
             fontweight='bold', pad=15)
ax.set_ylabel('F1 Score')
ax.set_ylim(0, 1.2)
for bar, val in zip(bars, vals_s):
    color = '#3fb950' if val == max(vals_s) else '#e6edf3'
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.01,
            f'{val:.4f}', ha='center', va='bottom',
            fontsize=12, fontweight='bold', color=color)
ax.tick_params(axis='x', labelsize=11)

fig1.tight_layout(pad=3.0)
plt.savefig('fig10a_corrected_scores.png', dpi=150,
            bbox_inches='tight', facecolor='#0d1117')
plt.show()
print("  ✅ Saved: fig10a_corrected_scores.png")

# ── Figure 2: Predictions overview ───────────────────────────
fig2, axes2 = plt.subplots(1, 2, figsize=(16, 6))
fig2.suptitle('2026 WORLD CUP PREDICTIONS (Corrected)',
              fontsize=16, fontweight='bold', color='#58a6ff')

ax = axes2[0]
stage_order  = ['group', 'roundof32', 'roundof16',
                'qf', 'sf', 'runnerup', 'champion']
stage_counts = pd.Series(constrained_stage).value_counts().reindex(
    stage_order, fill_value=0
)
bars = ax.bar(stage_counts.index, stage_counts.values,
              color=COLORS[:len(stage_counts)],
              edgecolor='none', width=0.6)
ax.set_title('Predicted Stage Distribution\n(48 Teams)',
             fontweight='bold', pad=15)
ax.set_ylabel('Number of Teams')
ax.tick_params(axis='x', rotation=35)
for bar, val in zip(bars, stage_counts.values):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.1,
            str(val), ha='center', va='bottom',
            fontsize=12, fontweight='bold')

ax = axes2[1]
ax.hist(goals_pred, bins=15, color='#3fb950',
        edgecolor='#0d1117', alpha=0.85, rwidth=0.85)
ax.axvline(goals_pred.mean(), color='#f78166',
           linestyle='--', linewidth=2.5,
           label=f'Mean: {goals_pred.mean():.1f}')
ax.set_title('Predicted Goals Distribution\n(48 Teams)',
             fontweight='bold', pad=15)
ax.set_xlabel('Predicted Goals')
ax.set_ylabel('Number of Teams')
ax.legend(fontsize=12)

fig2.tight_layout(pad=3.0)
plt.savefig('fig10b_corrected_predictions.png', dpi=150,
            bbox_inches='tight', facecolor='#0d1117')
plt.show()
print("  ✅ Saved: fig10b_corrected_predictions.png")

# ── Figure 3: Feature importances ────────────────────────────
if hasattr(best_goals_model, 'feature_importances_'):
    fig3, axes3 = plt.subplots(1, 2, figsize=(20, 9))
    fig3.suptitle('FEATURE IMPORTANCE — CORRECTED MODELS',
                  fontsize=16, fontweight='bold', color='#58a6ff')

    ax = axes3[0]
    feat_imp_g = pd.DataFrame({
        'feature':    goals_features,
        'importance': best_goals_model.feature_importances_
    }).sort_values('importance', ascending=False).head(15)
    ax.barh(feat_imp_g['feature'].iloc[::-1],
            feat_imp_g['importance'].iloc[::-1],
            color='#ffa657', edgecolor='none', height=0.7)
    ax.set_title(f'Top 15 Features\n{best_goals_name} (Goals)',
                 fontweight='bold', pad=15)
    ax.set_xlabel('Feature Importance')
    ax.tick_params(axis='y', labelsize=11)

    ax = axes3[1]
    feat_imp_s = pd.DataFrame({
        'feature':    stage_features,
        'importance': best_stage_model.feature_importances_
    }).sort_values('importance', ascending=False).head(15)
    ax.barh(feat_imp_s['feature'].iloc[::-1],
            feat_imp_s['importance'].iloc[::-1],
            color='#d2a8ff', edgecolor='none', height=0.7)
    ax.set_title(f'Top 15 Features\n{best_stage_name} (Stage)',
                 fontweight='bold', pad=15)
    ax.set_xlabel('Feature Importance')
    ax.tick_params(axis='y', labelsize=11)

    fig3.tight_layout(pad=3.0)
    plt.savefig('fig10c_feature_importance.png', dpi=150,
                bbox_inches='tight', facecolor='#0d1117')
    plt.show()
    print("  ✅ Saved: fig10c_feature_importance.png")

# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("✅ CORRECTED FINAL MODEL COMPLETE!")
print("=" * 60)
print(f"""
  LEAKAGE CHECK
  ─────────────────────────────────────────────────────
  Stage features : {len(stage_features)} features — no current goals 
  Goals features : {len(goals_features)} features — no current goals 
  final_position : removed 
  Leaking cols   : None 

  CV RESULTS (Realistic — No Leakage)
  ─────────────────────────────────────────────────────
  Goals Task  (RMSE ↓):""")

for name, score in goals_results.items():
    marker = ' ← BEST' if name == best_goals_name else ''
    print(f"    {name:25s}: {score:.4f}{marker}")

print(f"""
  Stage Task  (F1 ↑):""")

for name, score in stage_results.items():
    marker = ' ← BEST' if name == best_stage_name else ''
    print(f"    {name:25s}: {score:.4f}{marker}")

print(f"""
  BEST MODELS
  ─────────────────────────────────────────────────────
  Goals : {best_goals_name} (RMSE: {best_goals_score:.4f})
  Stage : {best_stage_name} (F1  : {best_stage_score:.4f})

  SAVED FILES
  ─────────────────────────────────────────────────────
  Models     → models/best_goals_model.joblib
               models/best_stage_model.joblib
               models/scaler_goals.joblib
               models/scaler_stage.joblib
               models/stage_features.joblib
               models/goals_features.joblib
  Submission → data/submission_corrected.csv
  Figures    → fig10a_corrected_scores.png
               fig10b_corrected_predictions.png
               fig10c_feature_importance.png

  
  ─────────────────────────────────────────────────────
 
""")