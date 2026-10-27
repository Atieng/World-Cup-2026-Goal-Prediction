"""Train the final Poisson model on all men's World Cups (1930-2022) and
predict 2026: total goals per team, plus stage reached via slot ranking."""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.features import (FEATURES, STAGE_ORDER, build_stages, features_for_next,
                          load_data, prior_features)
from src.train import assign_slots

# 2026 format: 48 teams -> 32 advance from groups
SLOTS_2026 = {6: 1, 5: 1, 4: 2, 3: 4, 2: 8, 1: 16, 0: 16}
STAGE_NAMES = {v: k for k, v in STAGE_ORDER.items()}

NAME_FIXES = {
    "Czechia": "Czech Republic",
    "Turkiye": "Turkey",
    "Cabo Verde": "Cape Verde",
    "Cote d'Ivoire": "Ivory Coast",
    "DR Congo": "Congo DR",
}


def map_team_ids(test, teams, train):
    """Match Test.csv countries to team_ids (NaN = no match found)."""
    if "team_id" in test.columns and test["team_id"].notna().all():
        return test["team_id"]
    lookup = teams.set_index("team_name")["team_id"].to_dict()
    lookup.update(train.drop_duplicates("country").set_index("country")["team_id"].to_dict())
    return test["country"].replace(NAME_FIXES).map(lookup)


if __name__ == "__main__":
    d = load_data("data")
    stages = build_stages(d["Train"], d["tournament_standings"])
    hist = prior_features(d["team_appearances"], d["tournaments"], stages)

    # 1. Final model on all history
    model = make_pipeline(StandardScaler(), PoissonRegressor(alpha=1.0, max_iter=1000))
    model.fit(hist[FEATURES], hist["total_goals"])

    # 2. 2026 features
    test = d["Test"]
    assert len(test) == 48 == sum(SLOTS_2026.values())
    team_ids = map_team_ids(test, d["teams"], d["Train"])
    unmatched = test.loc[team_ids.isna(), "country"].tolist()
    print(f"Matched {team_ids.notna().sum()}/48 teams. "
          f"Treated as debutants (no match): {unmatched or 'none'}")
    X26 = features_for_next(hist, team_ids.tolist())

    # 3. Predict goals, then rank teams into 2026 slots
    goals = model.predict(X26[FEATURES])
    stage = assign_slots(goals, SLOTS_2026)

    sub = pd.DataFrame({
        "ID": test["ID"],
        "total_goals": goals.round(2),
        "Target": [STAGE_NAMES[s] for s in stage],
    })

    out_dir = Path("submission-csv-files")
    out_dir.mkdir(exist_ok=True)
    sub.to_csv(out_dir / "submission_poisson_v2.csv", index=False)

    Path("models").mkdir(exist_ok=True)
    joblib.dump(model, "models/poisson_goals.joblib")

    preview = sub.assign(country=test["country"]).sort_values("total_goals", ascending=False)
    print("\nTop 12 predicted teams:")
    print(preview[["country", "total_goals", "Target"]].head(12).to_string(index=False))
    print("\nStage counts:")
    print(sub["Target"].value_counts().to_string())
    print(f"\nSaved: {out_dir / 'submission_poisson_v2.csv'}")
