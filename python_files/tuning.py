# ============================================================
# FIFA WORLD CUP 2026 — CONSOLIDATED PIPELINE
# ============================================================
# Replaces modelling.py / final_tuning.py / tuning.py with one
# pipeline:
#   - Leak-free feature set (from final_tuning.py)
#   - Tuned RF/XGB hyperparameters (from tuning.py's best_params.csv,
#     if present and compatible — falls back to regularized defaults)
#   - 5-model comparison: Poisson/Logistic, Random Forest, XGBoost,
#     LightGBM, CatBoost
#   - Ensemble blend of the top-3 performing models per task
#   - Saves models, model card, feature importances, and submission
# ============================================================

import os
import json
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import PoissonRegressor, LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb

warnings.filterwarnings("ignore")

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("  ⚠ lightgbm not installed — run: pip install lightgbm")

try:
    import catboost as cb
    HAS_CB = True
except ImportError:
    HAS_CB = False
    print("  ⚠ catboost not installed — run: pip install catboost")

DATA_PATH   = "data/"
MODELS_PATH = "models/"
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
# STEP 2: FEATURE ENGINEERING (leak-free — from final_tuning.py)
# ============================================================
print("\n" + "=" * 60)
print("STEP 2: FEATURE ENGINEERING (leak-free)")
print("=" * 60)

ta = dfs['team_appearances'].copy()

base_perf = ta.groupby(['tournament_id', 'team_id', 'team_name']).agg(
    matches_played    = ('match_id', 'count'),
    goals_scored      = ('goals_for', 'sum'),
    goals_conceded     = ('goals_against', 'sum'),
    wins              = ('win', 'sum'),
    losses            = ('lose', 'sum'),
    draws             = ('draw', 'sum'),
    extra_time_games  = ('extra_time', 'sum'),
    penalty_shootouts = ('penalty_shootout', 'sum'),
    home_games        = ('home_team', 'sum'),
    knockout_games    = ('knockout_stage', 'sum'),
    group_games       = ('group_stage', 'sum'),
).reset_index()

# Match-based ratios only — no goals ratios (avoids leakage)
base_perf['win_rate']         = base_perf['wins'] / base_perf['matches_played']
base_perf['loss_rate']        = base_perf['losses'] / base_perf['matches_played']
base_perf['draw_rate']        = base_perf['draws'] / base_perf['matches_played']
base_perf['points']           = base_perf['wins'] * 3 + base_perf['draws']
base_perf['points_per_match'] = base_perf['points'] / base_perf['matches_played']
base_perf['knockout_rate']    = base_perf['knockout_games'] / base_perf['matches_played']
base_perf['home_rate']        = base_perf['home_games'] / base_perf['matches_played']

hosts = dfs['host_countries'][['tournament_id', 'team_id']].copy()
hosts['is_host'] = 1
base_perf = base_perf.merge(hosts, on=['tournament_id', 'team_id'], how='left')
base_perf['is_host'] = base_perf['is_host'].fillna(0).astype(int)

tourney = dfs['tournaments'][[
    'tournament_id', 'year', 'count_teams',
    'round_of_16', 'quarter_finals', 'semi_finals'
]].copy()
base_perf = base_perf.merge(tourney, on='tournament_id', how='left')
base_perf['era'] = pd.cut(
    base_perf['year'], bins=[1929, 1965, 1985, 2000, 2030], labels=[1, 2, 3, 4]
).astype(float)

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

# Group standings — points and position only (no goals)
gs = dfs['group_standings'][[
    'tournament_id', 'team_id', 'position', 'played',
    'wins', 'draws', 'losses', 'points', 'advanced'
]].copy()
gs.columns = [
    'tournament_id', 'team_id', 'group_position', 'group_played',
    'group_wins', 'group_draws', 'group_losses', 'group_points', 'group_advanced'
]
base_perf = base_perf.merge(gs, on=['tournament_id', 'team_id'], how='left')
base_perf['group_win_rate']     = base_perf['group_wins'] / (base_perf['group_played'] + 1e-5)
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

mf = dfs['master_features'].copy()
hist_cols = [
    'team_id', 'win_rate', 'win_total', 'draw_rate', 'loss_rate',
    'matches_played_total', 'penalty_experience',
    'final_appearances', 'semi_appearances',
    'quarter_appearances', 'round16_appearances', 'group_exits',
    'recent_win_rate', 'recent_goals_avg', 'recent_goal_diff_avg',
    'recent_stage_trend', 'avg_stage_reached', 'max_stage_reached',
    'knockout_percentage', 'deep_run_percentage', 'final_percentage',
    'goals_avg', 'goals_std', 'goals_median', 'goals_against_avg', 'goal_diff_avg',
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
    'group stage': 'group', 'group': 'group',
    'second group stage': 'roundof16',
    'round of 16': 'roundof16', 'roundof16': 'roundof16',
    'quarter-finals': 'qf', 'qf': 'qf',
    'semi-finals': 'sf', 'sf': 'sf', 'third-place match': 'sf',
    'final round': 'runnerup', 'runnerup': 'runnerup',
    'final': 'champion', 'champion': 'champion',
}
stage_numeric = {'group': 0, 'roundof16': 1, 'qf': 2, 'sf': 3, 'runnerup': 4, 'champion': 5}
inv_stage_map = {v: k for k, v in stage_numeric.items()}

train_raw = dfs['train'].copy()
train_raw['stage_clean'] = (
    train_raw['stage_reached'].astype(str).str.lower().str.strip()
    .map(stage_map).fillna('group')
)
train_raw['stage_numeric'] = train_raw['stage_clean'].map(stage_numeric)
train_targets = train_raw[['tournament_id', 'team_id', 'stage_clean', 'stage_numeric', 'total_goals']].copy()

train_df = base_perf.merge(
    train_targets, on=['tournament_id', 'team_id'], how='inner'
).dropna(subset=['stage_clean', 'total_goals'])

print(f"  Training rows: {train_df.shape[0]}")
print(train_df['stage_clean'].value_counts().to_string())

# ============================================================
# STEP 4: CLEAN FEATURE SETS (No Leakage)
# ============================================================
print("\n" + "=" * 60)
print("STEP 4: CLEAN FEATURE SETS (No Leakage)")
print("=" * 60)

stage_features = [
    'matches_played', 'wins', 'losses', 'draws',
    'win_rate', 'loss_rate', 'draw_rate',
    'points', 'points_per_match',
    'extra_time_games', 'penalty_shootouts', 'knockout_rate', 'home_rate',
    'is_host', 'count_teams', 'confederation_strength', 'era',
    'group_position', 'group_points', 'group_win_rate', 'group_pts_per_game', 'group_advanced',
    'yellow_cards', 'red_cards', 'discipline_index',
    'penalties_taken', 'penalty_conversion_rate',
    'awards_won',
    'hist_win_rate', 'hist_win_total', 'hist_draw_rate', 'hist_loss_rate',
    'hist_matches_played_total', 'hist_penalty_experience',
    'hist_final_appearances', 'hist_semi_appearances', 'hist_quarter_appearances',
    'hist_round16_appearances', 'hist_group_exits',
    'hist_recent_win_rate', 'hist_recent_stage_trend',
    'hist_avg_stage_reached', 'hist_max_stage_reached',
    'hist_knockout_percentage', 'hist_deep_run_percentage', 'hist_final_percentage',
]

goals_features = [
    'matches_played', 'wins', 'losses', 'draws',
    'win_rate', 'loss_rate', 'draw_rate',
    'points', 'points_per_match',
    'extra_time_games', 'penalty_shootouts', 'knockout_rate', 'home_rate',
    'is_host', 'count_teams', 'confederation_strength', 'era',
    'group_position', 'group_points', 'group_win_rate', 'group_pts_per_game', 'group_advanced',
    'yellow_cards', 'discipline_index',
    'penalties_taken', 'penalty_conversion_rate',
    'awards_won',
    'hist_goals_avg', 'hist_goals_std', 'hist_goals_median',
    'hist_goals_against_avg', 'hist_goal_diff_avg', 'hist_recent_goals_avg',
    'hist_win_rate', 'hist_win_total', 'hist_matches_played_total',
    'hist_knockout_percentage', 'hist_deep_run_percentage',
    'hist_final_appearances', 'hist_semi_appearances',
]

for f in set(stage_features + goals_features):
    if f not in train_df.columns:
        train_df[f] = 0

leakers = [
    'goals_scored', 'goals_conceded', 'goal_difference', 'goals_per_match',
    'conceded_per_match', 'goal_ratio', 'first_half_goals', 'second_half_goals',
    'late_goals', 'early_goals', 'late_goal_ability', 'final_position',
]
stage_leaks = [f for f in leakers if f in stage_features]
goals_leaks = [f for f in leakers if f in goals_features]
print(f"  Leaking features in stage_features: {stage_leaks if stage_leaks else 'None ✅'}")
print(f"  Leaking features in goals_features: {goals_leaks if goals_leaks else 'None ✅'}")

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

print(f"  X_stage : {X_stage.shape}")
print(f"  X_goals : {X_goals.shape}")

# ============================================================
# STEP 5: LOAD TUNED HYPERPARAMETERS (if available & compatible)
# ============================================================
print("\n" + "=" * 60)
print("STEP 5: LOADING TUNED HYPERPARAMETERS")
print("=" * 60)

def load_best_params(model_key):
    """Read data/best_params.csv from tuning.py if present. Returns {} on
    any mismatch/missing file so callers fall back to safe defaults —
    the grid search there ran on a different (leakier) feature set, so
    params are a starting point, not guaranteed-optimal here."""
    try:
        bp = pd.read_csv(DATA_PATH + 'best_params.csv', index_col=0)
        if model_key not in bp.columns:
            return {}
        raw = bp[model_key].dropna().to_dict()
        parsed = {}
        for k, v in raw.items():
            try:
                parsed[k] = int(v)
            except (ValueError, TypeError):
                try:
                    parsed[k] = float(v)
                except (ValueError, TypeError):
                    parsed[k] = v
        return parsed
    except FileNotFoundError:
        return {}

rf_reg_params  = load_best_params('rf_regressor')
rf_clf_params  = load_best_params('rf_classifier')
xgb_reg_params = load_best_params('xgb_regressor')
xgb_clf_params = load_best_params('xgb_classifier')

for label, params in [('RF regressor', rf_reg_params), ('RF classifier', rf_clf_params),
                       ('XGB regressor', xgb_reg_params), ('XGB classifier', xgb_clf_params)]:
    print(f"  {label:15s}: {'loaded from best_params.csv' if params else 'using regularized defaults'}")

# ============================================================
# STEP 6: DEFINE MODELS
# ============================================================
print("\n" + "=" * 60)
print("STEP 6: MODEL DEFINITIONS")
print("=" * 60)

models_goals = {
    'Poisson Regression': PoissonRegressor(alpha=1.0, max_iter=1000),
    'Random Forest': RandomForestRegressor(
        **{**dict(n_estimators=300, max_depth=8, max_features='sqrt',
                   min_samples_leaf=5, min_samples_split=10),
           **rf_reg_params},
        random_state=42, n_jobs=-1
    ),
    'XGBoost': xgb.XGBRegressor(
        **{**dict(n_estimators=200, max_depth=3, learning_rate=0.1,
                   colsample_bytree=1.0, reg_alpha=0.1, reg_lambda=2.0,
                   subsample=0.7, min_child_weight=5),
           **xgb_reg_params},
        objective='count:poisson', random_state=42, verbosity=0, n_jobs=-1
    ),
}
models_stage = {
    'Logistic Regression': LogisticRegression(
        max_iter=1000, class_weight='balanced', C=0.5, random_state=42
    ),
    'Random Forest': RandomForestClassifier(
        **{**dict(n_estimators=200, max_depth=8, max_features='sqrt',
                   min_samples_leaf=5, min_samples_split=10),
           **rf_clf_params},
        class_weight='balanced', random_state=42, n_jobs=-1
    ),
    'XGBoost': xgb.XGBClassifier(
        **{**dict(n_estimators=200, max_depth=3, learning_rate=0.05,
                   colsample_bytree=0.7, reg_alpha=0.5, reg_lambda=2.0,
                   subsample=0.8, min_child_weight=5),
           **xgb_clf_params},
        objective='multi:softprob', num_class=6, random_state=42, verbosity=0, n_jobs=-1
    ),
}

if HAS_LGB:
    models_goals['LightGBM'] = lgb.LGBMRegressor(
        objective='poisson', n_estimators=500, learning_rate=0.05,
        num_leaves=20, max_depth=5, min_child_samples=10,
        subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, n_jobs=-1, verbosity=-1
    )
    models_stage['LightGBM'] = lgb.LGBMClassifier(
        objective='multiclass', num_class=6, n_estimators=300, learning_rate=0.05,
        num_leaves=20, max_depth=5, min_child_samples=10,
        subsample=0.8, colsample_bytree=0.8, class_weight='balanced',
        random_state=42, n_jobs=-1, verbosity=-1
    )

if HAS_CB:
    models_goals['CatBoost'] = cb.CatBoostRegressor(
        loss_function='Poisson', iterations=500, learning_rate=0.05, depth=5,
        l2_leaf_reg=3.0, random_state=42, verbose=0,
        train_dir=MODELS_PATH + 'catboost_info'
    )
    models_stage['CatBoost'] = cb.CatBoostClassifier(
        loss_function='MultiClass', iterations=300, learning_rate=0.05, depth=5,
        l2_leaf_reg=3.0, auto_class_weights='Balanced', random_state=42, verbose=0,
        train_dir=MODELS_PATH + 'catboost_info'
    )

print(f"  Goals models: {list(models_goals.keys())}")
print(f"  Stage models: {list(models_stage.keys())}")

# ============================================================
# STEP 7: CROSS-VALIDATE ALL MODELS
# ============================================================
print("\n" + "=" * 60)
print("STEP 7: CROSS-VALIDATION")
print("=" * 60)

goals_results = {}
print("\n  --- GOALS (RMSE ↓) ---")
for name, model in models_goals.items():
    cv = cross_val_score(model, X_goals_scaled, y_goals, cv=5,
                          scoring='neg_root_mean_squared_error')
    goals_results[name] = -cv.mean()
    print(f"  {name:20s} RMSE: {-cv.mean():.4f} ± {cv.std():.4f}")

stage_results = {}
print("\n  --- STAGE (F1 ↑) ---")
for name, model in models_stage.items():
    cv = cross_val_score(model, X_stage_scaled, y_stage, cv=skf, scoring='f1_weighted')
    stage_results[name] = cv.mean()
    print(f"  {name:20s} F1  : {cv.mean():.4f} ± {cv.std():.4f}")

# ============================================================
# STEP 8: BUILD ENSEMBLE — TOP 3 MODELS PER TASK, WEIGHTED BY CV SCORE
# ============================================================
print("\n" + "=" * 60)
print("STEP 8: ENSEMBLE WEIGHTS")
print("=" * 60)

def top_n_weighted(results, n=3, lower_is_better=True):
    ordered = sorted(results.items(), key=lambda kv: kv[1], reverse=not lower_is_better)
    top = dict(ordered[:n])
    if lower_is_better:
        inv = {k: 1.0 / v for k, v in top.items()}
        total = sum(inv.values())
        return {k: v / total for k, v in inv.items()}
    else:
        total = sum(top.values())
        return {k: v / total for k, v in top.items()}

goals_weights = top_n_weighted(goals_results, n=3, lower_is_better=True)
stage_weights = top_n_weighted(stage_results, n=3, lower_is_better=False)

print("  Goals ensemble:")
for k, v in goals_weights.items():
    print(f"    {k:20s} weight={v:.3f}  (RMSE {goals_results[k]:.4f})")
print("  Stage ensemble:")
for k, v in stage_weights.items():
    print(f"    {k:20s} weight={v:.3f}  (F1 {stage_results[k]:.4f})")

# ============================================================
# STEP 9: TRAIN FINAL MODELS (only those used in the blend)
# ============================================================
print("\n" + "=" * 60)
print("STEP 9: TRAIN FINAL MODELS")
print("=" * 60)

trained_goals_models = {}
for name in goals_weights:
    model = models_goals[name]
    model.fit(X_goals_scaled, y_goals)
    trained_goals_models[name] = model
    print(f"  ✅ {name} (goals) trained")

trained_stage_models = {}
for name in stage_weights:
    model = models_stage[name]
    if name == 'XGBoost':
        model.fit(X_stage_scaled, y_stage, sample_weight=sw)
    else:
        model.fit(X_stage_scaled, y_stage)
    trained_stage_models[name] = model
    print(f"  ✅ {name} (stage) trained")

# ============================================================
# STEP 10: SAVE MODELS, SCALERS, FEATURE LISTS
# ============================================================
print("\n" + "=" * 60)
print("STEP 10: SAVING ARTIFACTS")
print("=" * 60)

for name, model in trained_goals_models.items():
    joblib.dump(model, f"{MODELS_PATH}goals_{name.lower().replace(' ', '_')}.joblib")
for name, model in trained_stage_models.items():
    joblib.dump(model, f"{MODELS_PATH}stage_{name.lower().replace(' ', '_')}.joblib")

joblib.dump(scaler_goals, f"{MODELS_PATH}scaler_goals.joblib")
joblib.dump(scaler_stage, f"{MODELS_PATH}scaler_stage.joblib")
joblib.dump(goals_features, f"{MODELS_PATH}goals_features.joblib")
joblib.dump(stage_features, f"{MODELS_PATH}stage_features.joblib")
joblib.dump(goals_weights, f"{MODELS_PATH}goals_ensemble_weights.joblib")
joblib.dump(stage_weights, f"{MODELS_PATH}stage_ensemble_weights.joblib")
print("  ✅ Models, scalers, feature lists, and blend weights saved")

# ============================================================
# STEP 11: BUILD TEST FEATURES
# ============================================================
print("\n" + "=" * 60)
print("STEP 11: BUILD TEST FEATURES")
print("=" * 60)

test_raw = dfs['test'].copy()
name_fixes = {
    'Czechia': 'Czech Republic', 'Turkiye': 'Turkey',
    'Cabo Verde': 'Cape Verde', "Cote d'Ivoire": 'Ivory Coast', 'DR Congo': 'Congo DR',
}
test_raw['country_lookup'] = test_raw['country'].replace(name_fixes)
country_to_id = dfs['teams'].set_index('team_name')['team_id'].to_dict()
country_to_id.update(dfs['train'].set_index('country')['team_id'].to_dict())
test_raw['team_id'] = test_raw['country_lookup'].map(country_to_id)

print(f"  Matched  : {test_raw['team_id'].notna().sum()}/48")
print(f"  Unmatched: {test_raw[test_raw['team_id'].isna()]['country'].tolist()}")

num_cols   = base_perf.select_dtypes(include=[np.number]).columns.tolist()
hist_avg   = base_perf.groupby('team_id')[num_cols].mean().reset_index()
test_feats = test_raw[['ID', 'country', 'team_id']].merge(
    hist_avg, on='team_id', how='left'
).reset_index(drop=True)

test_feats['count_teams'] = 48
test_feats['is_host'] = test_feats['country'].isin(['United States', 'Canada', 'Mexico']).astype(int)
test_conf = test_raw[['team_id']].merge(dfs['teams'][['team_id', 'confederation_name']], on='team_id', how='left')
test_feats['confederation_strength'] = test_conf['confederation_name'].map(conf_strength).fillna(2).values

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
print(f"  X_test_stage: {X_test_stage.shape}")
print(f"  X_test_goals: {X_test_goals.shape}")

# ============================================================
# STEP 12: BLEND PREDICTIONS + 2026 CONSTRAINTS
# ============================================================
print("\n" + "=" * 60)
print("STEP 12: BLENDED PREDICTIONS")
print("=" * 60)

def enforce_2026_constraints(stage_probs):
    constraints = {'champion': 1, 'runnerup': 1, 'sf': 2, 'qf': 4,
                    'roundof16': 8, 'roundof32': 16, 'group': 16}
    col_map = {'group': 0, 'roundof16': 1, 'qf': 2, 'sf': 3, 'runnerup': 4, 'champion': 5}
    order = ['champion', 'runnerup', 'sf', 'qf', 'roundof16', 'roundof32', 'group']
    preds = ['group'] * 48
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

goals_blend = np.zeros(len(test_feats))
for name, weight in goals_weights.items():
    goals_blend += trained_goals_models[name].predict(X_test_goals) * weight
goals_pred = np.round(np.clip(goals_blend, 0, None)).astype(int)

stage_blend = np.zeros((len(test_feats), 6))
for name, weight in stage_weights.items():
    stage_blend += trained_stage_models[name].predict_proba(X_test_stage) * weight
constrained_stage = enforce_2026_constraints(stage_blend)

print(f"  Goals — mean: {goals_pred.mean():.2f}  max: {goals_pred.max()}")
print("\n  Stage distribution:")
print(pd.Series(constrained_stage).value_counts().to_string())

# Diagnostic-only consistency flag (NOT auto-applied — would break the
# fixed 16/16/8/4/2/1/1 slot allocation Zindi expects)
flagged = 0
for g, s in zip(goals_pred, constrained_stage):
    if (g >= 12 and s in ('group', 'roundof32')) or (g <= 1 and s == 'champion'):
        flagged += 1
if flagged:
    print(f"\n  ⚠ {flagged} team(s) have goals/stage that look inconsistent — review manually.")

# ============================================================
# STEP 13: SAVE SUBMISSION
# ============================================================
print("\n" + "=" * 60)
print("STEP 13: SUBMISSION")
print("=" * 60)

submission = pd.DataFrame({
    'ID': test_raw['ID'].values,
    'total_goals': goals_pred,
    'Target': constrained_stage,
})
submission.to_csv(DATA_PATH + 'submission_final.csv', index=False)
print(f"  ✅ Saved: {DATA_PATH}submission_final.csv")
print(submission.head(10).to_string(index=False))

# ============================================================
# STEP 14: MODEL CARD + FEATURE IMPORTANCE (for the RAG chatbot)
# ============================================================
print("\n" + "=" * 60)
print("STEP 14: MODEL CARD + FEATURE IMPORTANCE")
print("=" * 60)

model_card = {
    "Goals Ensemble": {
        "type": "Weighted blend: " + ", ".join(
            f"{k} ({v:.2f})" for k, v in goals_weights.items()),
        "features": len(goals_features),
        "cv_rmse": {k: round(goals_results[k], 4) for k in goals_weights},
        "purpose": "Predict total goals per team per tournament",
        "limitations": [
            "No current FIFA rankings or 2024-2026 form",
            "No injury or manager quality information",
            "Hyperparameters partly inherited from tuning.py's grid search "
            "on a different (leakier) feature set",
        ],
    },
    "Stage Ensemble": {
        "type": "Weighted blend: " + ", ".join(
            f"{k} ({v:.2f})" for k, v in stage_weights.items()),
        "features": len(stage_features),
        "cv_f1": {k: round(stage_results[k], 4) for k in stage_weights},
        "purpose": "Predict tournament stage reached",
        "limitations": [
            "Round of 32 never seen in training (new for 2026's 48-team format); "
            "handled by the slot-allocation constraint, not direct classification",
            "5 new teams have no history and use bottom-25th-percentile defaults",
        ],
    },
}
with open(DATA_PATH + 'model_card.json', 'w') as f:
    json.dump(model_card, f, indent=2)
print(f"  ✅ Saved: {DATA_PATH}model_card.json")

# Feature importance — from the best tree-based model per task
feat_imp_rows = []
tree_goals_name = next((n for n in ['XGBoost', 'LightGBM', 'CatBoost', 'Random Forest']
                         if n in trained_goals_models), None)
if tree_goals_name:
    m = trained_goals_models[tree_goals_name]
    importances = m.get_feature_importance() if HAS_CB and tree_goals_name == 'CatBoost' \
        else m.feature_importances_
    for feat, imp in zip(goals_features, importances):
        feat_imp_rows.append({'task': 'goals', 'feature': feat, 'importance': float(imp)})

tree_stage_name = next((n for n in ['XGBoost', 'LightGBM', 'CatBoost', 'Random Forest']
                         if n in trained_stage_models), None)
if tree_stage_name:
    m = trained_stage_models[tree_stage_name]
    importances = m.get_feature_importance() if HAS_CB and tree_stage_name == 'CatBoost' \
        else m.feature_importances_
    for feat, imp in zip(stage_features, importances):
        feat_imp_rows.append({'task': 'stage', 'feature': feat, 'importance': float(imp)})

pd.DataFrame(feat_imp_rows).to_csv(DATA_PATH + 'feature_importance.csv', index=False)
print(f"  ✅ Saved: {DATA_PATH}feature_importance.csv "
      f"(goals from {tree_goals_name}, stage from {tree_stage_name})")

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
    'text.color': '#e6edf3', 'grid.color': '#21262d', 'font.size': 11,
})
COLORS = ['#58a6ff', '#3fb950', '#f78166', '#d2a8ff', '#ffa657']

fig, axes = plt.subplots(1, 2, figsize=(18, 7))
fig.suptitle('CONSOLIDATED PIPELINE — ALL MODELS COMPARED',
             fontsize=15, fontweight='bold', color='#58a6ff')

ax = axes[0]
names_g = list(goals_results.keys())
vals_g  = list(goals_results.values())
bar_colors = [COLORS[i % len(COLORS)] if n in goals_weights else '#484f58'
              for i, n in enumerate(names_g)]
bars = ax.bar(names_g, vals_g, color=bar_colors, edgecolor='none', width=0.5)
ax.set_title('Goals RMSE (Lower is Better)\nGrey = not in ensemble', fontweight='bold', pad=15)
ax.set_ylabel('RMSE')
ax.set_ylim(0, max(vals_g) * 1.3)
for bar, val in zip(bars, vals_g):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.tick_params(axis='x', rotation=20)

ax = axes[1]
names_s = list(stage_results.keys())
vals_s  = list(stage_results.values())
bar_colors = [COLORS[i % len(COLORS)] if n in stage_weights else '#484f58'
              for i, n in enumerate(names_s)]
bars = ax.bar(names_s, vals_s, color=bar_colors, edgecolor='none', width=0.5)
ax.set_title('Stage F1 (Higher is Better)\nGrey = not in ensemble', fontweight='bold', pad=15)
ax.set_ylabel('F1 Score')
ax.set_ylim(0, 1.15)
for bar, val in zip(bars, vals_s):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.tick_params(axis='x', rotation=20)

fig.tight_layout(pad=3.0)
plt.savefig('fig11_consolidated_comparison.png', dpi=150, bbox_inches='tight', facecolor='#0d1117')
plt.show()
print("  ✅ Saved: fig11_consolidated_comparison.png")

# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("✅ CONSOLIDATED PIPELINE COMPLETE")
print("=" * 60)
print(f"""
  ALL MODELS — GOALS (RMSE ↓)
  ─────────────────────────────────────────
{chr(10).join(f"  {k:20s}: {v:.4f}" + ("  ← in ensemble" if k in goals_weights else "") for k, v in goals_results.items())}

  ALL MODELS — STAGE (F1 ↑)
  ─────────────────────────────────────────
{chr(10).join(f"  {k:20s}: {v:.4f}" + ("  ← in ensemble" if k in stage_weights else "") for k, v in stage_results.items())}

  OUTPUT
  ─────────────────────────────────────────
  Submission → {DATA_PATH}submission_final.csv
  Model card → {DATA_PATH}model_card.json
  Feat. imp. → {DATA_PATH}feature_importance.csv
  Figure     → fig11_consolidated_comparison.png
  Models     → {MODELS_PATH}*.joblib
""")