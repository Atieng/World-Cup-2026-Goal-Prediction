"""Point-in-time features for World Cup goal and stage prediction.

Every feature for a (team, tournament) row uses only tournaments
that happened BEFORE it, so training rows look exactly like the
2026 rows the model will predict on.
"""
from pathlib import Path

import numpy as np
import pandas as pd

# One shared stage mapping for the whole project
STAGE_MAP = {
    "group stage":        "group",
    "second group stage": "qf",        # 1974/78/82 second round ≈ last 8–12
    "round of 16":        "roundof16",
    "quarter-finals":     "qf",
    "semi-finals":        "sf",
    "third-place match":  "sf",
    "final round":        "sf",        # 1950 final group; top 2 fixed below
    "final":              "runnerup",  # champion fixed below
}
STAGE_ORDER = {"group": 0, "roundof32": 1, "roundof16": 2, "qf": 3,
               "sf": 4, "runnerup": 5, "champion": 6}

FEATURES = ["prev_apps", "first_appearance", "prev_gf_pm", "prev_ga_pm",
            "prev_win_rate", "last_gf", "last_stage", "best_stage",
            "years_since"]


def load_data(data_dir="data"):
    d = Path(data_dir)
    return {name: pd.read_csv(d / f"{name}.csv") for name in
            ["team_appearances", "tournaments", "tournament_standings",
             "teams", "Train", "Test"]}


def build_stages(train, standings):
    """Stage reached per (tournament, team), with champion/runner-up
    taken from tournament_standings."""
    s = train[["tournament_id", "team_id", "stage_reached", "total_goals"]].copy()
    s["stage"] = s["stage_reached"].str.lower().str.strip().map(STAGE_MAP)

    unmapped = s.loc[s["stage"].isna(), "stage_reached"].unique()
    assert len(unmapped) == 0, f"Unmapped stages: {unmapped}"

    pos = standings[["tournament_id", "team_id", "position"]]
    s = s.merge(pos, on=["tournament_id", "team_id"], how="left")
    s.loc[s["position"] == 1, "stage"] = "champion"
    s.loc[s["position"] == 2, "stage"] = "runnerup"

    s["stage_numeric"] = s["stage"].map(STAGE_ORDER)
    return s.drop(columns="position")


def prior_features(ta, tournaments, stages):
    """One row per (team, tournament), using only earlier tournaments."""
    mens = tournaments[tournaments["tournament_name"].str.contains("Men's", na=False)]
    per = (ta[ta["tournament_id"].isin(mens["tournament_id"])]
           .groupby(["tournament_id", "team_id"])
           .agg(gf=("goals_for", "sum"), ga=("goals_against", "sum"),
                w=("win", "sum"), n=("match_id", "count"))
           .reset_index()
           .merge(mens[["tournament_id", "year"]], on="tournament_id")
           .merge(stages[["tournament_id", "team_id", "stage_numeric", "total_goals"]],
                  on=["tournament_id", "team_id"])
           .sort_values(["team_id", "year"]))

    g = per.groupby("team_id")
    prev_n = g["n"].cumsum() - per["n"]
    per["prev_apps"]     = g.cumcount()
    per["prev_gf_pm"]    = (g["gf"].cumsum() - per["gf"]) / prev_n.replace(0, np.nan)
    per["prev_ga_pm"]    = (g["ga"].cumsum() - per["ga"]) / prev_n.replace(0, np.nan)
    per["prev_win_rate"] = (g["w"].cumsum()  - per["w"])  / prev_n.replace(0, np.nan)
    per["last_gf"]       = g["gf"].shift(1)
    per["last_stage"]    = g["stage_numeric"].shift(1)
    per["best_stage"]    = g["stage_numeric"].transform(lambda s: s.shift(1).cummax())
    per["years_since"]   = per["year"] - g["year"].shift(1)
    return _fill_first_appearances(per)


def features_for_next(per, team_ids, year=2026):
    """Same features for an upcoming tournament, built from ALL past rows."""
    hist = per.sort_values("year")
    g = hist.groupby("team_id")
    out = g.agg(prev_apps=("gf", "size"), gf=("gf", "sum"), ga=("ga", "sum"),
                w=("w", "sum"), n=("n", "sum"),
                best_stage=("stage_numeric", "max"), last_year=("year", "max"))
    last = g.tail(1).set_index("team_id")

    out["prev_gf_pm"]    = out["gf"] / out["n"]
    out["prev_ga_pm"]    = out["ga"] / out["n"]
    out["prev_win_rate"] = out["w"] / out["n"]
    out["last_gf"]       = last["gf"]
    out["last_stage"]    = last["stage_numeric"]
    out["years_since"]   = year - out["last_year"]

    # Teams with no history (debutants) get NaN, then filled below
    out = out.reindex(team_ids).reset_index().rename(columns={"index": "team_id"})
    out["prev_apps"] = out["prev_apps"].fillna(0)
    return _fill_first_appearances(out)


def _fill_first_appearances(df):
    df["first_appearance"] = (df["prev_apps"] == 0).astype(int)
    df["prev_gf_pm"]    = df["prev_gf_pm"].fillna(0)
    df["prev_ga_pm"]    = df["prev_ga_pm"].fillna(0)
    df["prev_win_rate"] = df["prev_win_rate"].fillna(0)
    df["last_gf"]       = df["last_gf"].fillna(0)
    df["last_stage"]    = df["last_stage"].fillna(-1)   # -1 = never played
    df["best_stage"]    = df["best_stage"].fillna(-1)
    df["years_since"]   = df["years_since"].fillna(99)
    return df