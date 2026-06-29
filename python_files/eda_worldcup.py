"""
=============================================================
FIFA WORLD CUP 2026 - COMPREHENSIVE DATA ANALYSIS
=============================================================
Covers: Data Understanding, Cleaning & Visualization
Datasets: team_appearances, qualified_teams, group_standings,
          goals, matches, master_features, award_winners,
          bookings, host_countries, penalty_kicks,
          tournament_standings, tournaments, teams,
          Train.csv, Test.csv
=============================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ── Plot Styling ─────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#0d1117',
    'axes.facecolor':   '#161b22',
    'axes.edgecolor':   '#30363d',
    'axes.labelcolor':  '#e6edf3',
    'xtick.color':      '#8b949e',
    'ytick.color':      '#8b949e',
    'text.color':       '#e6edf3',
    'grid.color':       '#21262d',
    'grid.linestyle':   '--',
    'grid.alpha':       0.5,
    'font.family':      'DejaVu Sans',
    'font.size':        11,
})

COLORS = ['#58a6ff', '#3fb950', '#f78166', '#d2a8ff',
          '#ffa657', '#79c0ff', '#56d364', '#ff7b72']

import os

# ── Auto-detect data folder path ─────────────────────────────
# All CSV files including Train.csv and Test.csv are in data/
# Adjust this path if your folder is named differently
POSSIBLE_PATHS = ['data/', '../data/', './', '../']
DATA_PATH = None
for p in POSSIBLE_PATHS:
    if os.path.exists(p + 'Train.csv') or os.path.exists(p + 'train.csv'):
        DATA_PATH = p
        break

if DATA_PATH is None:
    DATA_PATH = 'data/'  # fallback default
    
print(f"📁 Using data path: {DATA_PATH}")

# =============================================================
# SECTION 0: LOAD ALL DATASETS
# =============================================================
print("=" * 60)
print("LOADING ALL DATASETS")
print("=" * 60)

# All files are in the SAME folder
datasets = {
    # ✅ Already using
    'team_appearances':     'team_appearances.csv',
    'host_countries':       'host_countries.csv',
    'award_winners':        'award_winners.csv',
    'bookings':             'bookings.csv',
    'penalty_kicks':        'penalty_kicks.csv',
    'tournament_standings': 'tournament_standings.csv',
    'tournaments':          'tournaments.csv',
    'teams':                'teams.csv',
    # ❗ Critical / Important
    'qualified_teams':      'qualified_teams.csv',
    'group_standings':      'group_standings.csv',
    'goals':                'goals.csv',
    'matches':              'matches.csv',
    'master_features':      'master_features.csv',
    # 🎯 Challenge files — same folder as all others
    'train':                'Train.csv',
    'test':                 'Test.csv',
}

dfs = {}
for key, fname in datasets.items():
    try:
        path = DATA_PATH + fname
        dfs[key] = pd.read_csv(path)
        print(f"  ✅ {key:25s} → {dfs[key].shape}")
    except Exception as e:
        print(f"  ❌ {key:25s} → ERROR: {e}")

# =============================================================
# SECTION 1: DATA UNDERSTANDING
# =============================================================
print("\n" + "=" * 60)
print("SECTION 1: DATA UNDERSTANDING")
print("=" * 60)

def profile_dataset(name, df):
    print(f"\n{'─'*55}")
    print(f"📋 DATASET: {name.upper()}")
    print(f"{'─'*55}")
    print(f"  Shape        : {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"  Memory       : {df.memory_usage(deep=True).sum() / 1024:.1f} KB")
    print(f"  Duplicates   : {df.duplicated().sum()}")
    print(f"\n  COLUMNS & DTYPES:")
    for col in df.columns:
        nulls = df[col].isna().sum()
        pct   = nulls / len(df) * 100
        dtype = str(df[col].dtype)
        null_flag = f"  ⚠️  {nulls:,} nulls ({pct:.1f}%)" if nulls > 0 else ""
        print(f"    • {col:35s} [{dtype:10s}]{null_flag}")
    print(f"\n  SAMPLE (first 2 rows):")
    print(df.head(2).to_string())

for name, df in dfs.items():
    profile_dataset(name, df)

# =============================================================
# SECTION 2: DATA CLEANING
# =============================================================
print("\n" + "=" * 60)
print("SECTION 2: DATA CLEANING")
print("=" * 60)

# ── 2.1 Train Data ───────────────────────────────────────────
print("\n[Train.csv]")
train = dfs['train'].copy()
print(f"  Before: {train.shape}")
train.drop_duplicates(inplace=True)
print(f"  After dedup: {train.shape}")

# Standardise stage labels
stage_map = {
    'group stage':        'group',
    'group':              'group',
    'second group stage': 'roundof16',
    'round of 16':        'roundof16',
    'roundof16':          'roundof16',
    'quarter-finals':     'qf',
    'qf':                 'qf',
    'semi-finals':        'sf',
    'sf':                 'sf',
    'third-place match':  'sf',
    'final round':        'runnerup',
    'runnerup':           'runnerup',
    'final':              'champion',
    'champion':           'champion',
}
if 'stage_reached' in train.columns:
    train['stage_clean'] = (train['stage_reached']
                            .astype(str).str.lower().str.strip()
                            .map(stage_map).fillna('group'))
    print(f"  Stage distribution:\n{train['stage_clean'].value_counts().to_string()}")

# ── 2.2 Team Appearances ─────────────────────────────────────
print("\n[team_appearances]")
ta = dfs['team_appearances'].copy()
print(f"  Nulls:\n{ta.isnull().sum()[ta.isnull().sum() > 0].to_string()}")
ta.fillna(0, inplace=True)
print(f"  After fill: {ta.isnull().sum().sum()} nulls remain")

# ── 2.3 Goals ────────────────────────────────────────────────
print("\n[goals]")
g = dfs['goals'].copy()
print(f"  Nulls:\n{g.isnull().sum()[g.isnull().sum() > 0].to_string()}")
if 'minute_label' in g.columns:
    g['minute_label'] = g['minute_label'].fillna('unknown')
if 'match_period' in g.columns:
    g['match_period'] = g['match_period'].fillna('unknown')

# ── 2.4 Matches ──────────────────────────────────────────────
print("\n[matches]")
m = dfs['matches'].copy()
print(f"  Nulls:\n{m.isnull().sum()[m.isnull().sum() > 0].to_string()}")

# ── 2.5 Group Standings ──────────────────────────────────────
print("\n[group_standings]")
gs = dfs['group_standings'].copy()
print(f"  Nulls:\n{gs.isnull().sum()[gs.isnull().sum() > 0].to_string()}")
gs.fillna(0, inplace=True)

# ── 2.6 Qualified Teams ──────────────────────────────────────
print("\n[qualified_teams]")
qt = dfs['qualified_teams'].copy()
print(f"  Nulls:\n{qt.isnull().sum()[qt.isnull().sum() > 0].to_string()}")

# ── 2.7 Master Features ──────────────────────────────────────
print("\n[master_features]")
mf = dfs['master_features'].copy()
null_cols = mf.isnull().sum()[mf.isnull().sum() > 0]
print(f"  Null columns: {len(null_cols)}")
mf.fillna(mf.median(numeric_only=True), inplace=True)
print(f"  After fill: {mf.isnull().sum().sum()} nulls remain")

# ── 2.8 Bookings ─────────────────────────────────────────────
print("\n[bookings]")
bk = dfs['bookings'].copy()
print(f"  Nulls:\n{bk.isnull().sum()[bk.isnull().sum() > 0].to_string()}")

print("\n✅ Data cleaning complete!")

# =============================================================
# SECTION 3: DATA VISUALIZATION
# =============================================================
print("\n" + "=" * 60)
print("SECTION 3: DATA VISUALIZATION")
print("=" * 60)

# ─────────────────────────────────────────────────────────────
# FIGURE 1: TRAIN DATA OVERVIEW
# ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('TRAIN DATA OVERVIEW', fontsize=16, fontweight='bold',
             color='#58a6ff', y=1.01)

# 1a. Stage distribution
if 'stage_clean' in train.columns:
    ax = axes[0, 0]
    stage_counts = train['stage_clean'].value_counts()
    bars = ax.bar(stage_counts.index, stage_counts.values,
                  color=COLORS[:len(stage_counts)], edgecolor='none')
    ax.set_title('Stage Distribution', fontweight='bold')
    ax.set_xlabel('Stage')
    ax.set_ylabel('Count')
    ax.tick_params(axis='x', rotation=30)
    for bar, val in zip(bars, stage_counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                str(val), ha='center', va='bottom', fontsize=9)

# 1b. Goals distribution
if 'total_goals' in train.columns:
    ax = axes[0, 1]
    ax.hist(train['total_goals'].dropna(), bins=20,
            color='#3fb950', edgecolor='#0d1117', alpha=0.85)
    ax.axvline(train['total_goals'].mean(), color='#f78166',
               linestyle='--', linewidth=2, label=f"Mean: {train['total_goals'].mean():.1f}")
    ax.set_title('Total Goals Distribution', fontweight='bold')
    ax.set_xlabel('Total Goals')
    ax.set_ylabel('Frequency')
    ax.legend()

# 1c. Goals by stage
if 'stage_clean' in train.columns and 'total_goals' in train.columns:
    ax = axes[0, 2]
    stage_goals = train.groupby('stage_clean')['total_goals'].mean().sort_values(ascending=False)
    bars = ax.bar(stage_goals.index, stage_goals.values,
                  color=COLORS[:len(stage_goals)], edgecolor='none')
    ax.set_title('Avg Goals by Stage', fontweight='bold')
    ax.set_xlabel('Stage')
    ax.set_ylabel('Avg Goals')
    ax.tick_params(axis='x', rotation=30)
    for bar, val in zip(bars, stage_goals.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{val:.1f}', ha='center', va='bottom', fontsize=9)

# 1d. Teams per year
if 'year' in train.columns:
    ax = axes[1, 0]
    teams_per_year = train.groupby('year')['team_id'].nunique() if 'team_id' in train.columns \
                     else train.groupby('year').size()
    ax.plot(teams_per_year.index, teams_per_year.values,
            color='#d2a8ff', marker='o', linewidth=2, markersize=6)
    ax.set_title('Teams per Tournament Year', fontweight='bold')
    ax.set_xlabel('Year')
    ax.set_ylabel('Number of Teams')
    ax.grid(True, alpha=0.3)

# 1e. Top goal-scoring teams
if 'total_goals' in train.columns and 'country' in train.columns:
    ax = axes[1, 1]
    top_teams = train.groupby('country')['total_goals'].sum().nlargest(10)
    ax.barh(top_teams.index[::-1], top_teams.values[::-1],
            color='#ffa657', edgecolor='none')
    ax.set_title('Top 10 Goal-Scoring Teams (All Time)', fontweight='bold')
    ax.set_xlabel('Total Goals')

# 1f. Confederation distribution
if 'confederation_name' in train.columns:
    ax = axes[1, 2]
    conf_counts = train['confederation_name'].value_counts()
    wedges, texts, autotexts = ax.pie(
        conf_counts.values,
        labels=conf_counts.index,
        colors=COLORS[:len(conf_counts)],
        autopct='%1.1f%%',
        startangle=90
    )
    for text in autotexts:
        text.set_color('#0d1117')
        text.set_fontsize(8)
    ax.set_title('Teams by Confederation', fontweight='bold')

plt.tight_layout()
plt.savefig('fig1_train_overview.png', dpi=150, bbox_inches='tight',
            facecolor='#0d1117')
plt.show()
print("  ✅ Figure 1 saved: fig1_train_overview.png")

# ─────────────────────────────────────────────────────────────
# FIGURE 2: TEAM APPEARANCES ANALYSIS
# ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('TEAM APPEARANCES ANALYSIS', fontsize=16,
             fontweight='bold', color='#58a6ff')

ta_agg = ta.groupby('team_id').agg(
    total_goals=('goals_for', 'sum'),
    total_conceded=('goals_against', 'sum'),
    total_wins=('win', 'sum'),
    total_matches=('match_id', 'count'),
    avg_win_rate=('win', 'mean')
).reset_index()

# 2a. Goals scored vs conceded
ax = axes[0, 0]
ax.scatter(ta_agg['total_goals'], ta_agg['total_conceded'],
           alpha=0.6, color='#58a6ff', edgecolors='none', s=60)
ax.plot([0, ta_agg['total_goals'].max()],
        [0, ta_agg['total_goals'].max()],
        color='#f78166', linestyle='--', alpha=0.7, label='Equal line')
ax.set_title('Goals Scored vs Conceded', fontweight='bold')
ax.set_xlabel('Goals Scored')
ax.set_ylabel('Goals Conceded')
ax.legend()

# 2b. Win rate distribution
ax = axes[0, 1]
ax.hist(ta_agg['avg_win_rate'].dropna(), bins=20,
        color='#3fb950', edgecolor='#0d1117', alpha=0.85)
ax.axvline(ta_agg['avg_win_rate'].mean(), color='#f78166',
           linestyle='--', linewidth=2,
           label=f"Mean: {ta_agg['avg_win_rate'].mean():.2f}")
ax.set_title('Team Win Rate Distribution', fontweight='bold')
ax.set_xlabel('Win Rate')
ax.set_ylabel('Count')
ax.legend()

# 2c. Goals per match over years
if 'year' in ta.columns:
    ax = axes[0, 2]
    yearly = ta.merge(dfs['tournaments'][['tournament_id', 'year']],
                      on='tournament_id', how='left')
    yearly_goals = yearly.groupby('year')['goals_for'].mean()
    ax.plot(yearly_goals.index, yearly_goals.values,
            color='#ffa657', marker='o', linewidth=2, markersize=5)
    ax.fill_between(yearly_goals.index, yearly_goals.values,
                    alpha=0.2, color='#ffa657')
    ax.set_title('Avg Goals Per Match Over Years', fontweight='bold')
    ax.set_xlabel('Year')
    ax.set_ylabel('Avg Goals')
    ax.grid(True, alpha=0.3)

# 2d. Top 15 teams by wins
ax = axes[1, 0]
top_wins = ta_agg.nlargest(15, 'total_wins')
if 'team_name' in ta.columns:
    team_names = ta[['team_id', 'team_name']].drop_duplicates()
    top_wins = top_wins.merge(team_names, on='team_id', how='left')
    ax.barh(top_wins['team_name'][::-1], top_wins['total_wins'][::-1],
            color='#58a6ff', edgecolor='none')
else:
    ax.barh(range(len(top_wins)), top_wins['total_wins'].values[::-1],
            color='#58a6ff', edgecolor='none')
ax.set_title('Top 15 Teams by Total Wins', fontweight='bold')
ax.set_xlabel('Total Wins')

# 2e. Matches played distribution
ax = axes[1, 1]
ax.hist(ta_agg['total_matches'], bins=15,
        color='#d2a8ff', edgecolor='#0d1117', alpha=0.85)
ax.set_title('Matches Played per Team', fontweight='bold')
ax.set_xlabel('Total Matches')
ax.set_ylabel('Number of Teams')

# 2f. Win rate vs goals scored
ax = axes[1, 2]
sc = ax.scatter(ta_agg['avg_win_rate'], ta_agg['total_goals'],
                c=ta_agg['total_matches'], cmap='YlOrRd',
                alpha=0.7, s=60, edgecolors='none')
plt.colorbar(sc, ax=ax, label='Total Matches')
ax.set_title('Win Rate vs Goals (colored by Matches)', fontweight='bold')
ax.set_xlabel('Win Rate')
ax.set_ylabel('Total Goals')

plt.tight_layout()
plt.savefig('fig2_team_appearances.png', dpi=150, bbox_inches='tight',
            facecolor='#0d1117')
plt.show()
print("  ✅ Figure 2 saved: fig2_team_appearances.png")

# ─────────────────────────────────────────────────────────────
# FIGURE 3: GOALS ANALYSIS
# ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('GOALS DEEP ANALYSIS', fontsize=16,
             fontweight='bold', color='#58a6ff')

# 3a. Goals by match period
if 'match_period' in g.columns:
    ax = axes[0, 0]
    period_counts = g['match_period'].value_counts()
    bars = ax.bar(period_counts.index, period_counts.values,
                  color=COLORS[:len(period_counts)], edgecolor='none')
    ax.set_title('Goals by Match Period', fontweight='bold')
    ax.set_xlabel('Period')
    ax.set_ylabel('Number of Goals')
    ax.tick_params(axis='x', rotation=30)

# 3b. Goal types
if 'goal_type' in g.columns:
    ax = axes[0, 1]
    goal_types = g['goal_type'].value_counts()
    wedges, texts, autotexts = ax.pie(
        goal_types.values,
        labels=goal_types.index,
        colors=COLORS[:len(goal_types)],
        autopct='%1.1f%%', startangle=90
    )
    for t in autotexts:
        t.set_color('#0d1117')
        t.set_fontsize(8)
    ax.set_title('Goal Types Breakdown', fontweight='bold')

# 3c. Goals per tournament over time
if 'tournament_id' in g.columns:
    ax = axes[0, 2]
    tourney_goals = g.groupby('tournament_id').size().reset_index(name='goals')
    tourney_goals = tourney_goals.merge(
        dfs['tournaments'][['tournament_id', 'year']], on='tournament_id', how='left'
    ).sort_values('year')
    ax.bar(tourney_goals['year'].astype(str), tourney_goals['goals'],
           color='#3fb950', edgecolor='none', alpha=0.85)
    ax.set_title('Total Goals per Tournament', fontweight='bold')
    ax.set_xlabel('Year')
    ax.set_ylabel('Total Goals')
    ax.tick_params(axis='x', rotation=45)

# 3d. Top goal-scoring teams all time
if 'team_id' in g.columns and 'team_name' in g.columns:
    ax = axes[1, 0]
    top_scorers = g.groupby('team_name').size().nlargest(12)
    ax.barh(top_scorers.index[::-1], top_scorers.values[::-1],
            color='#ffa657', edgecolor='none')
    ax.set_title('Top 12 Goal-Scoring Teams', fontweight='bold')
    ax.set_xlabel('Goals Scored')

# 3e. Own goals vs regular goals
if 'own_goal' in g.columns:
    ax = axes[1, 1]
    own_goal_counts = g['own_goal'].value_counts()
    labels = ['Regular Goals', 'Own Goals']
    ax.bar(labels, own_goal_counts.values,
           color=['#3fb950', '#f78166'], edgecolor='none')
    ax.set_title('Regular vs Own Goals', fontweight='bold')
    ax.set_ylabel('Count')
    for i, v in enumerate(own_goal_counts.values):
        ax.text(i, v + 5, str(v), ha='center', fontsize=10)

# 3f. Goals by confederation
if 'confederation_id' in g.columns or 'team_id' in g.columns:
    ax = axes[1, 2]
    goals_with_conf = g.merge(
        dfs['teams'][['team_id', 'confederation_name']],
        on='team_id', how='left'
    )
    conf_goals = goals_with_conf['confederation_name'].value_counts()
    bars = ax.bar(conf_goals.index, conf_goals.values,
                  color=COLORS[:len(conf_goals)], edgecolor='none')
    ax.set_title('Goals by Confederation', fontweight='bold')
    ax.set_xlabel('Confederation')
    ax.set_ylabel('Goals')
    ax.tick_params(axis='x', rotation=30)

plt.tight_layout()
plt.savefig('fig3_goals_analysis.png', dpi=150, bbox_inches='tight',
            facecolor='#0d1117')
plt.show()
print("  ✅ Figure 3 saved: fig3_goals_analysis.png")

# ─────────────────────────────────────────────────────────────
# FIGURE 4: MATCHES & TOURNAMENT ANALYSIS
# ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('MATCHES & TOURNAMENT ANALYSIS', fontsize=16,
             fontweight='bold', color='#58a6ff')

# 4a. Home vs Away wins
if 'home_team_win' in m.columns or 'result' in m.columns:
    ax = axes[0, 0]
    if 'home_team_win' in m.columns:
        home_wins = m['home_team_win'].sum()
        away_wins = len(m) - m['home_team_win'].sum() - m.get('draw', pd.Series([0])).sum()
        draws     = m.get('draw', pd.Series([0])).sum()
        labels    = ['Home Win', 'Away Win', 'Draw']
        values    = [home_wins, away_wins, draws]
    else:
        result_counts = m['result'].value_counts() if 'result' in m.columns else pd.Series()
        labels = result_counts.index.tolist()
        values = result_counts.values.tolist()
    ax.bar(labels, values, color=['#3fb950', '#f78166', '#ffa657'],
           edgecolor='none')
    ax.set_title('Match Outcomes', fontweight='bold')
    ax.set_ylabel('Number of Matches')

# 4b. Matches per tournament
if 'tournament_id' in m.columns:
    ax = axes[0, 1]
    matches_per = m.groupby('tournament_id').size().reset_index(name='matches')
    matches_per = matches_per.merge(
        dfs['tournaments'][['tournament_id', 'year']], on='tournament_id', how='left'
    ).sort_values('year')
    ax.plot(matches_per['year'], matches_per['matches'],
            color='#58a6ff', marker='o', linewidth=2, markersize=5)
    ax.fill_between(matches_per['year'], matches_per['matches'],
                    alpha=0.2, color='#58a6ff')
    ax.set_title('Matches per Tournament Over Time', fontweight='bold')
    ax.set_xlabel('Year')
    ax.set_ylabel('Number of Matches')
    ax.grid(True, alpha=0.3)

# 4c. Tournament size over time
ax = axes[0, 2]
t_df = dfs['tournaments'].sort_values('year') if 'year' in dfs['tournaments'].columns \
       else dfs['tournaments']
if 'count_teams' in t_df.columns and 'year' in t_df.columns:
    ax.step(t_df['year'], t_df['count_teams'],
            color='#d2a8ff', linewidth=2, where='post')
    ax.scatter(t_df['year'], t_df['count_teams'],
               color='#d2a8ff', s=60, zorder=5)
    ax.set_title('Tournament Size Over Time', fontweight='bold')
    ax.set_xlabel('Year')
    ax.set_ylabel('Number of Teams')
    ax.grid(True, alpha=0.3)
    # Annotate 2026
    ax.axhline(48, color='#f78166', linestyle='--', alpha=0.7, label='2026: 48 teams')
    ax.legend()

# 4d. Group standings — points distribution
if 'points' in gs.columns:
    ax = axes[1, 0]
    ax.hist(gs['points'].dropna(), bins=15,
            color='#56d364', edgecolor='#0d1117', alpha=0.85)
    ax.axvline(gs['points'].mean(), color='#f78166', linestyle='--',
               linewidth=2, label=f"Mean: {gs['points'].mean():.1f}")
    ax.set_title('Group Stage Points Distribution', fontweight='bold')
    ax.set_xlabel('Points')
    ax.set_ylabel('Frequency')
    ax.legend()

# 4e. Qualified teams — stage reached distribution
if 'stage_reached' in qt.columns:
    ax = axes[1, 1]
    qt_stages = qt['stage_reached'].value_counts()
    bars = ax.bar(qt_stages.index, qt_stages.values,
                  color=COLORS[:len(qt_stages)], edgecolor='none')
    ax.set_title('Qualified Teams — Stage Reached', fontweight='bold')
    ax.set_xlabel('Stage')
    ax.set_ylabel('Count')
    ax.tick_params(axis='x', rotation=30)
    for bar, val in zip(bars, qt_stages.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                str(val), ha='center', va='bottom', fontsize=9)

# 4f. Host country performance
if 'is_host' in ta.columns or len(dfs['host_countries']) > 0:
    ax = axes[1, 2]
    hc = dfs['host_countries'].copy()
    if 'team_id' in hc.columns and 'performance' in hc.columns:
        perf_counts = hc['performance'].value_counts()
        ax.bar(perf_counts.index, perf_counts.values,
               color='#ffa657', edgecolor='none')
        ax.set_title('Host Country Performance', fontweight='bold')
        ax.set_xlabel('Stage Reached')
        ax.set_ylabel('Count')
        ax.tick_params(axis='x', rotation=30)
    else:
        host_merge = ta.merge(
            dfs['host_countries'][['tournament_id', 'team_id']].assign(is_host=1),
            on=['tournament_id', 'team_id'], how='left'
        )
        host_merge['is_host'] = host_merge['is_host'].fillna(0)
        host_avg = host_merge.groupby('is_host')['win'].mean()
        ax.bar(['Non-Host', 'Host'], host_avg.values,
               color=['#58a6ff', '#ffa657'], edgecolor='none')
        ax.set_title('Win Rate: Host vs Non-Host', fontweight='bold')
        ax.set_ylabel('Average Win Rate')
        for i, v in enumerate(host_avg.values):
            ax.text(i, v + 0.005, f'{v:.3f}', ha='center', fontsize=10)

plt.tight_layout()
plt.savefig('fig4_matches_tournament.png', dpi=150, bbox_inches='tight',
            facecolor='#0d1117')
plt.show()
print("  ✅ Figure 4 saved: fig4_matches_tournament.png")

# ─────────────────────────────────────────────────────────────
# FIGURE 5: BOOKINGS & DISCIPLINE ANALYSIS
# ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('DISCIPLINE & BOOKINGS ANALYSIS', fontsize=16,
             fontweight='bold', color='#58a6ff')

# 5a. Bookings per tournament over time
if 'tournament_id' in bk.columns:
    ax = axes[0, 0]
    bk_tourney = bk.groupby('tournament_id').size().reset_index(name='bookings')
    bk_tourney = bk_tourney.merge(
        dfs['tournaments'][['tournament_id', 'year']], on='tournament_id', how='left'
    ).sort_values('year')
    ax.bar(bk_tourney['year'].astype(str), bk_tourney['bookings'],
           color='#f78166', edgecolor='none', alpha=0.85)
    ax.set_title('Total Bookings per Tournament', fontweight='bold')
    ax.set_xlabel('Year')
    ax.set_ylabel('Bookings')
    ax.tick_params(axis='x', rotation=45)

# 5b. Yellow vs Red cards
if 'sending_off' in bk.columns:
    ax = axes[0, 1]
    card_types = bk['sending_off'].value_counts()
    labels = ['Yellow Card', 'Red Card']
    values = [card_types.get(0, 0), card_types.get(1, 0)]
    ax.bar(labels, values, color=['#ffa657', '#f78166'], edgecolor='none')
    ax.set_title('Yellow vs Red Cards', fontweight='bold')
    ax.set_ylabel('Count')
    for i, v in enumerate(values):
        ax.text(i, v + 5, f'{v:,}', ha='center', fontsize=10)

# 5c. Most booked teams
if 'team_name' in bk.columns:
    ax = axes[1, 0]
    top_booked = bk['team_name'].value_counts().nlargest(12)
    ax.barh(top_booked.index[::-1], top_booked.values[::-1],
            color='#ff7b72', edgecolor='none')
    ax.set_title('Most Disciplinary Issues (Top 12 Teams)', fontweight='bold')
    ax.set_xlabel('Total Bookings')

# 5d. Penalty kicks analysis
pk = dfs['penalty_kicks'].copy()
if 'converted' in pk.columns or 'scored' in pk.columns:
    ax = axes[1, 1]
    score_col = 'converted' if 'converted' in pk.columns else 'scored'
    pk_rate = pk[score_col].value_counts()
    labels = ['Scored', 'Missed']
    ax.pie([pk_rate.get(1, 0), pk_rate.get(0, 0)],
           labels=labels, colors=['#3fb950', '#f78166'],
           autopct='%1.1f%%', startangle=90)
    ax.set_title('Penalty Kick Conversion Rate', fontweight='bold')

plt.tight_layout()
plt.savefig('fig5_discipline.png', dpi=150, bbox_inches='tight',
            facecolor='#0d1117')
plt.show()
print("  ✅ Figure 5 saved: fig5_discipline.png")

# ─────────────────────────────────────────────────────────────
# FIGURE 6: MASTER FEATURES & CORRELATION
# ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(18, 8))
fig.suptitle('MASTER FEATURES ANALYSIS', fontsize=16,
             fontweight='bold', color='#58a6ff')

numeric_cols = mf.select_dtypes(include=[np.number]).columns.tolist()

# 6a. Correlation heatmap (top features)
ax = axes[0]
top_corr_cols = numeric_cols[:18] if len(numeric_cols) >= 18 else numeric_cols
corr = mf[top_corr_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, ax=ax, cmap='RdYlGn', center=0,
            annot=False, linewidths=0.3, cbar_kws={'shrink': 0.8})
ax.set_title('Feature Correlation Matrix', fontweight='bold')
ax.tick_params(axis='x', rotation=45, labelsize=8)
ax.tick_params(axis='y', rotation=0, labelsize=8)

# 6b. Top features by variance (most informative)
ax = axes[1]
feat_variance = mf[numeric_cols].std().sort_values(ascending=False).head(15)
ax.barh(feat_variance.index[::-1], feat_variance.values[::-1],
        color=COLORS * 3, edgecolor='none')
ax.set_title('Top 15 Features by Std Dev (Most Varied)', fontweight='bold')
ax.set_xlabel('Standard Deviation')

plt.tight_layout()
plt.savefig('fig6_master_features.png', dpi=150, bbox_inches='tight',
            facecolor='#0d1117')
plt.show()
print("  ✅ Figure 6 saved: fig6_master_features.png")

# ─────────────────────────────────────────────────────────────
# FIGURE 7: TEST SET & 2026 PREVIEW
# ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('2026 WORLD CUP — TEST SET PREVIEW', fontsize=16,
             fontweight='bold', color='#58a6ff')

test = dfs['test'].copy()  # loaded as lowercase 'test' key

# 7a. Test teams with historical coverage
test_with_hist = test.merge(
    mf[['team_name'] + numeric_cols[:5]] if 'team_name' in mf.columns else mf,
    left_on='country', right_on='team_name', how='left'
)
coverage = test_with_hist.iloc[:, 2].notna() if len(test_with_hist.columns) > 2 \
           else pd.Series([True] * len(test))

ax = axes[0]
matched   = coverage.sum()
unmatched = len(test) - matched
ax.bar(['Has History', 'No History'],
       [matched, unmatched],
       color=['#3fb950', '#f78166'], edgecolor='none')
ax.set_title('Test Teams — Historical Data Coverage', fontweight='bold')
ax.set_ylabel('Number of Teams')
for i, v in enumerate([matched, unmatched]):
    ax.text(i, v + 0.2, str(int(v)), ha='center', fontsize=12, fontweight='bold')

# 7b. 2026 teams by confederation
if 'country' in test.columns:
    test_conf = test.merge(
        dfs['teams'][['team_name', 'confederation_name']],
        left_on='country', right_on='team_name', how='left'
    )
    ax = axes[1]
    if 'confederation_name' in test_conf.columns:
        conf_dist = test_conf['confederation_name'].value_counts()
        bars = ax.bar(conf_dist.index, conf_dist.values,
                      color=COLORS[:len(conf_dist)], edgecolor='none')
        ax.set_title('2026 Teams by Confederation', fontweight='bold')
        ax.set_xlabel('Confederation')
        ax.set_ylabel('Number of Teams')
        ax.tick_params(axis='x', rotation=30)
        for bar, val in zip(bars, conf_dist.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    str(val), ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('fig7_test_preview.png', dpi=150, bbox_inches='tight',
            facecolor='#0d1117')
plt.show()
print("  ✅ Figure 7 saved: fig7_test_preview.png")

# =============================================================
# SECTION 4: SUMMARY REPORT
# =============================================================
print("\n" + "=" * 60)
print("SECTION 4: KEY INSIGHTS SUMMARY")
print("=" * 60)

print("""
📊 DATA OVERVIEW
────────────────────────────────────────
""")
for name, df in dfs.items():
    print(f"  {name:25s}: {df.shape[0]:,} rows × {df.shape[1]} cols")

print("""
🔍 KEY FINDINGS
────────────────────────────────────────
1. STAGE IMBALANCE
   - 'group' has most samples (~50%)
   - 'runnerup' only 4 samples → SMOTE or weighting needed

2. GOALS DISTRIBUTION
   - Right-skewed → Poisson regression is correct choice
   - Champions score significantly more than group-stage teams

3. HOST ADVANTAGE
   - Host teams historically outperform expectations
   - 3 hosts in 2026: USA, Canada, Mexico

4. NEW 2026 ROUND
   - Round of 32 is brand new — no historical data
   - Must be inferred from team strength scores

5. TEST SET COVERAGE
   - ~40/48 teams have historical World Cup data
   - 8 new teams need fallback feature imputation

6. FEATURE IMPORTANCE (Expected)
   - win_rate, goals_avg, hist_avg_win_rate → strongest
   - confederation, is_host, tournaments_played → moderate
   - bookings, penalty experience → weak but useful

📁 OUTPUT FILES
────────────────────────────────────────
  fig1_train_overview.png
  fig2_team_appearances.png
  fig3_goals_analysis.png
  fig4_matches_tournament.png
  fig5_discipline.png
  fig6_master_features.png
  fig7_test_preview.png
""")

print("🏁 EDA COMPLETE! Starting Feature Engineering...")

# =============================================================
# SECTION 5: FEATURE ENGINEERING
# Built directly from dfs{} already loaded in EDA pipeline
# =============================================================
print("\n" + "=" * 60)
print("SECTION 5: FEATURE ENGINEERING")
print("=" * 60)

# ── 5.1 BASE TEAM PERFORMANCE PER TOURNAMENT ─────────────────
print("\n[5.1] Base Performance Features...")

ta = dfs['team_appearances'].copy()

base_perf = ta.groupby(['tournament_id', 'team_id', 'team_name']).agg(
    matches_played        = ('match_id', 'count'),
    goals_scored          = ('goals_for', 'sum'),
    goals_conceded        = ('goals_against', 'sum'),
    wins                  = ('win', 'sum'),
    losses                = ('lose', 'sum'),
    draws                 = ('draw', 'sum'),
    extra_time_games      = ('extra_time', 'sum'),
    penalty_shootouts     = ('penalty_shootout', 'sum'),
    home_games            = ('home_team', 'sum'),
    away_games            = ('away_team', 'sum'),
    knockout_games        = ('knockout_stage', 'sum'),
    group_games           = ('group_stage', 'sum'),
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

print(f"  ✅ Base performance: {base_perf.shape}")

# ── 5.2 HOST COUNTRY FEATURE ─────────────────────────────────
print("\n[5.2] Host Country Features...")

hosts = dfs['host_countries'][['tournament_id', 'team_id']].copy()
hosts['is_host'] = 1
base_perf = base_perf.merge(hosts, on=['tournament_id', 'team_id'], how='left')
base_perf['is_host'] = base_perf['is_host'].fillna(0).astype(int)
print(f"  ✅ Host entries: {hosts.shape[0]}")

# ── 5.3 TOURNAMENT METADATA ───────────────────────────────────
print("\n[5.3] Tournament Metadata...")

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

print(f"  ✅ After tournament merge: {base_perf.shape}")

# ── 5.4 CONFEDERATION FEATURES ───────────────────────────────
print("\n[5.4] Confederation Features...")

teams_info = dfs['teams'][['team_id', 'confederation_name', 'region_name']]
base_perf  = base_perf.merge(teams_info, on='team_id', how='left')

conf_strength = {
    'Union of European Football Associations':                                      5,
    'South American Football Confederation':                                        5,
    'Confederation of North, Central American and Caribbean Association Football':  3,
    'Confederation of African Football':                                            3,
    'Asian Football Confederation':                                                 2,
    'Oceania Football Confederation':                                               1,
}
base_perf['confederation_strength'] = (
    base_perf['confederation_name'].map(conf_strength).fillna(2)
)
print(f"  ✅ After confederation merge: {base_perf.shape}")

# ── 5.5 GROUP STAGE FEATURES ─────────────────────────────────
print("\n[5.5] Group Stage Features...")

gs = dfs['group_standings'][[
    'tournament_id', 'team_id', 'position', 'played',
    'wins', 'draws', 'losses', 'goals_for',
    'goals_against', 'goal_difference', 'points', 'advanced'
]].copy()
gs.columns = [
    'tournament_id', 'team_id', 'group_position', 'group_played',
    'group_wins', 'group_draws', 'group_losses', 'group_goals_for',
    'group_goals_against', 'group_goal_diff', 'group_points', 'group_advanced'
]

base_perf = base_perf.merge(gs, on=['tournament_id', 'team_id'], how='left')
base_perf['group_win_rate']     = base_perf['group_wins'] / (base_perf['group_played'] + 1e-5)
base_perf['group_goals_ratio']  = (base_perf['group_goals_for'] + 1) / (base_perf['group_goals_against'] + 1)
base_perf['group_pts_per_game'] = base_perf['group_points'] / (base_perf['group_played'] + 1e-5)

print(f"  ✅ After group standings: {base_perf.shape}")

# ── 5.6 AWARD FEATURES ───────────────────────────────────────
print("\n[5.6] Award Features...")

awards = dfs['award_winners'].groupby(
    ['tournament_id', 'team_id']
).size().reset_index(name='awards_won')

base_perf = base_perf.merge(awards, on=['tournament_id', 'team_id'], how='left')
base_perf['awards_won'] = base_perf['awards_won'].fillna(0)
print(f"  ✅ After awards: {base_perf.shape}")

# ── 5.7 DISCIPLINE FEATURES ──────────────────────────────────
print("\n[5.7] Discipline Features...")

bk = dfs['bookings'].groupby(['tournament_id', 'team_id']).agg(
    yellow_cards = ('yellow_card', 'sum'),
    red_cards    = ('red_card', 'sum'),
    sending_offs = ('sending_off', 'sum'),
).reset_index()
bk['total_bookings']   = bk['yellow_cards'] + bk['red_cards']
bk['discipline_index'] = bk['yellow_cards'] + (bk['red_cards'] * 3) + (bk['sending_offs'] * 2)

base_perf = base_perf.merge(bk, on=['tournament_id', 'team_id'], how='left')
base_perf[['yellow_cards', 'red_cards', 'sending_offs',
           'total_bookings', 'discipline_index']] = \
    base_perf[['yellow_cards', 'red_cards', 'sending_offs',
               'total_bookings', 'discipline_index']].fillna(0)
print(f"  ✅ After bookings: {base_perf.shape}")

# ── 5.8 PENALTY FEATURES ─────────────────────────────────────
print("\n[5.8] Penalty Features...")

pk = dfs['penalty_kicks'].groupby(['tournament_id', 'team_id']).agg(
    penalties_taken  = ('converted', 'count'),
    penalties_scored = ('converted', 'sum'),
).reset_index()
pk['penalty_conversion_rate'] = pk['penalties_scored'] / (pk['penalties_taken'] + 1e-5)

base_perf = base_perf.merge(pk, on=['tournament_id', 'team_id'], how='left')
base_perf[['penalties_taken', 'penalties_scored',
           'penalty_conversion_rate']] = \
    base_perf[['penalties_taken', 'penalties_scored',
               'penalty_conversion_rate']].fillna(0)
print(f"  ✅ After penalties: {base_perf.shape}")

# ── 5.9 GOALS DETAIL FEATURES ────────────────────────────────
print("\n[5.9] Goals Detail Features...")

g = dfs['goals'].copy()
goals_detail = g.groupby(['tournament_id', 'team_id']).agg(
    first_half_goals  = ('match_period', lambda x: (x == 'first half').sum()),
    second_half_goals = ('match_period', lambda x: (x == 'second half').sum()),
    extra_time_goals  = ('match_period', lambda x: x.str.contains('extra', na=False).sum()),
    own_goals_scored  = ('own_goal', 'sum'),
    penalty_goals     = ('penalty', 'sum'),
    early_goals       = ('minute_regulation', lambda x: (x <= 15).sum()),
    late_goals        = ('minute_regulation', lambda x: (x >= 75).sum()),
).reset_index()

goals_detail['late_goal_ability'] = goals_detail['late_goals'] / (
    goals_detail['first_half_goals'] + goals_detail['second_half_goals'] + 1e-5
)
goals_detail['penalty_goal_rate'] = goals_detail['penalty_goals'] / (
    goals_detail['first_half_goals'] + goals_detail['second_half_goals'] + 1e-5
)

base_perf = base_perf.merge(goals_detail, on=['tournament_id', 'team_id'], how='left')
goal_detail_cols = [c for c in goals_detail.columns if c not in ['tournament_id', 'team_id']]
base_perf[goal_detail_cols] = base_perf[goal_detail_cols].fillna(0)
print(f"  ✅ After goals detail: {base_perf.shape}")

# ── 5.10 HISTORICAL MASTER FEATURES ──────────────────────────
print("\n[5.10] Historical Master Features...")

mf       = dfs['master_features'].copy()
hist_cols = [
    'team_id',
    'win_rate', 'win_total', 'draw_rate', 'loss_rate',
    'matches_played_total', 'penalty_experience',
    'final_appearances', 'semi_appearances',
    'quarter_appearances', 'round16_appearances', 'group_exits',
    'recent_win_rate', 'recent_goals_avg',
    'recent_goal_diff_avg', 'recent_stage_trend',
    'recent_final_appearances', 'recent_semi_appearances',
    'avg_stage_reached', 'max_stage_reached',
    'knockout_percentage', 'deep_run_percentage', 'final_percentage',
    'goals_avg', 'goals_std', 'goals_median',
    'goals_against_avg', 'goal_diff_avg',
]

mf_selected          = mf[hist_cols].copy()
mf_selected.columns  = ['team_id'] + ['hist_' + c for c in hist_cols[1:]]
base_perf            = base_perf.merge(mf_selected, on='team_id', how='left')
hist_feature_cols    = [c for c in mf_selected.columns if c != 'team_id']
base_perf[hist_feature_cols] = base_perf[hist_feature_cols].fillna(0)
print(f"  ✅ After master features: {base_perf.shape}")

# ── 5.11 TOURNAMENT STANDINGS ────────────────────────────────
print("\n[5.11] Tournament Standings...")

standings = dfs['tournament_standings'][['tournament_id', 'team_id', 'position']].copy()
standings.columns = ['tournament_id', 'team_id', 'final_position']
base_perf = base_perf.merge(standings, on=['tournament_id', 'team_id'], how='left')
base_perf['final_position'] = base_perf['final_position'].fillna(99)
print(f"  ✅ After standings: {base_perf.shape}")

# =============================================================
# SECTION 6: TARGET MAPPING & COMBINE DATASETS
# =============================================================
print("\n" + "=" * 60)
print("SECTION 6: TARGET MAPPING & COMBINE DATASETS")
print("=" * 60)

stage_map = {
    'group stage':        'group',
    'group':              'group',
    'second group stage': 'roundof16',
    'round of 16':        'roundof16',
    'roundof16':          'roundof16',
    'quarter-finals':     'qf',
    'qf':                 'qf',
    'semi-finals':        'sf',
    'sf':                 'sf',
    'third-place match':  'sf',
    'final round':        'runnerup',
    'runnerup':           'runnerup',
    'final':              'champion',
    'champion':           'champion',
}
stage_numeric = {
    'group': 0, 'roundof16': 1, 'qf': 2,
    'sf': 3,    'runnerup': 4,  'champion': 5
}

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

# Combine base_perf + targets
train_final = base_perf.merge(
    train_targets, on=['tournament_id', 'team_id'], how='inner'
).dropna(subset=['stage_clean', 'total_goals'])

print(f"\n  ✅ Combined training dataset: {train_final.shape}")
print(f"\n  Stage distribution:")
print(train_final['stage_clean'].value_counts().to_string())
print(f"\n  Goals stats:")
print(train_final['total_goals'].describe().round(2).to_string())

# =============================================================
# SECTION 7: BUILD TEST FEATURES
# =============================================================
print("\n" + "=" * 60)
print("SECTION 7: BUILD TEST FEATURES")
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
extra_map     = dfs['train'].set_index('country')['team_id'].to_dict()
country_to_id.update(extra_map)
test_raw['team_id'] = test_raw['country_lookup'].map(country_to_id)

print(f"  Matched  : {test_raw['team_id'].notna().sum()}/48")
print(f"  Unmatched: {test_raw[test_raw['team_id'].isna()]['country'].tolist()}")

# Historical averages per team
numeric_base_cols = base_perf.select_dtypes(include=[np.number]).columns.tolist()
hist_avg  = base_perf.groupby('team_id')[numeric_base_cols].mean().reset_index()
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

# Fill new teams with bottom 25%
bottom_25 = train_final.select_dtypes(include=[np.number]).quantile(0.25)
for col in test_feats.select_dtypes(include=[np.number]).columns:
    if col in bottom_25.index:
        test_feats[col] = test_feats[col].fillna(bottom_25[col])
    else:
        test_feats[col] = test_feats[col].fillna(0)

print(f"\n  ✅ Test features shape: {test_feats.shape}")

# =============================================================
# SECTION 8: DEFINE FEATURE SETS & SAVE
# =============================================================
print("\n" + "=" * 60)
print("SECTION 8: FEATURE SETS & SAVE")
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
    'awards_won',
    'yellow_cards', 'red_cards', 'discipline_index',
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
    'awards_won',
    'yellow_cards', 'discipline_index',
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

# Ensure all features exist
for f in set(stage_features + goals_features):
    if f not in train_final.columns:
        train_final[f] = 0
    if f not in test_feats.columns:
        test_feats[f]  = 0

# Final matrices — exposed globally for modelling notebook
X_stage      = train_final[stage_features].fillna(0).values
X_goals      = train_final[goals_features].fillna(0).values
y_stage      = train_final['stage_numeric'].values.astype(int)
y_goals      = train_final['total_goals'].values
X_test_stage = test_feats[stage_features].fillna(0).values
X_test_goals = test_feats[goals_features].fillna(0).values

# Save
train_final.to_csv('data/train_engineered.csv', index=False)
test_feats.to_csv('data/test_engineered.csv',   index=False)

print(f"  X_stage      : {X_stage.shape}")
print(f"  X_goals      : {X_goals.shape}")
print(f"  X_test_stage : {X_test_stage.shape}")
print(f"  X_test_goals : {X_test_goals.shape}")
print(f"\n  Stage distribution:")
for k, v in sorted(pd.Series(y_stage).value_counts().items()):
    print(f"    Class {k} → {v} samples")
print(f"\n  Goals — mean: {y_goals.mean():.2f}  std: {y_goals.std():.2f}")
print(f"\n  Stage features : {len(stage_features)}")
print(f"  Goals features : {len(goals_features)}")
print(f"\n  Saved → data/train_engineered.csv")
print(f"  Saved → data/test_engineered.csv")

print("\n" + "=" * 60)
print("✅ FULL PIPELINE COMPLETE!")
print("=" * 60)
print("""
  
""")