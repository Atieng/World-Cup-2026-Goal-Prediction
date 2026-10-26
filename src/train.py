"""Leave-one-tournament-out evaluation: train on earlier World Cups,
test on each recent one, and compare models against simple baselines."""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.metrics import f1_score, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.features import FEATURES, build_stages, load_data, prior_features

TEST_YEARS = [2010, 2014, 2018, 2022]

GOAL_MODELS = {
    "Poisson regression": lambda: make_pipeline(
        StandardScaler(), PoissonRegressor(alpha=1.0, max_iter=1000)),
    "Gradient boosting": lambda: HistGradientBoostingRegressor(
        loss="poisson", max_depth=3, learning_rate=0.05, max_iter=200,
        min_samples_leaf=10, random_state=42),
}
STAGE_MODELS = {
    "Logistic regression": lambda: make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")),
    "Gradient boosting": lambda: HistGradientBoostingClassifier(
        max_depth=3, learning_rate=0.05, max_iter=200, min_samples_leaf=10,
        class_weight="balanced", random_state=42),
}


def rmse(y, p):
    return float(np.sqrt(mean_squared_error(y, p)))


def macro_f1(y, p):
    return float(f1_score(y, p, average="macro", zero_division=0))


def assign_slots(score, slot_counts):
    """Rank teams by score (highest = strongest) and fill stages in order.
    slot_counts maps stage_numeric -> number of teams that finish there."""
    order = np.argsort(-np.asarray(score, dtype=float), kind="stable")
    pred = np.empty(len(order), dtype=int)
    i = 0
    for stage in sorted(slot_counts, reverse=True):
        n = slot_counts[stage]
        pred[order[i:i + n]] = stage
        i += n
    return pred


def evaluate(df):
    rows = []
    for year in TEST_YEARS:
        tr, va = df[df["year"] < year], df[df["year"] == year]
        X_tr, X_va = tr[FEATURES], va[FEATURES]
        y_g, y_s = va["total_goals"], va["stage_numeric"]
        mean_goals = tr["total_goals"].mean()
        slots = y_s.value_counts().to_dict()   # tournament format is known in advance

        # --- Goals baselines ---
        rows.append((year, "goals", "Baseline: train mean", rmse(y_g, np.full(len(va), mean_goals))))
        last = np.where(va["first_appearance"] == 1, mean_goals, va["last_gf"])
        rows.append((year, "goals", "Baseline: last tournament", rmse(y_g, last)))

        # --- Stage baselines ---
        rows.append((year, "stage", "Baseline: always group", macro_f1(y_s, np.zeros(len(va)))))
        rows.append((year, "stage", "Baseline: repeat last stage",
                     macro_f1(y_s, va["last_stage"].clip(lower=0))))
        rows.append((year, "stage", "Baseline: slots by past win rate",
                     macro_f1(y_s, assign_slots(va["prev_win_rate"], slots))))

        # --- Goal models (and their predictions used as a strength ranking) ---
        for name, make in GOAL_MODELS.items():
            pred_goals = make().fit(X_tr, tr["total_goals"]).predict(X_va)
            rows.append((year, "goals", name, rmse(y_g, pred_goals)))
            rows.append((year, "stage", f"{name} (goals) + slots",
                         macro_f1(y_s, assign_slots(pred_goals, slots))))

        # --- Stage models: raw prediction vs. expected stage + slots ---
        for name, make in STAGE_MODELS.items():
            m = make().fit(X_tr, tr["stage_numeric"])
            rows.append((year, "stage", name, macro_f1(y_s, m.predict(X_va))))
            expected = m.predict_proba(X_va) @ m.classes_
            rows.append((year, "stage", f"{name} + slots",
                         macro_f1(y_s, assign_slots(expected, slots))))

    return pd.DataFrame(rows, columns=["year", "task", "model", "score"])


if __name__ == "__main__":
    d = load_data("data")
    stages = build_stages(d["Train"], d["tournament_standings"])
    df = prior_features(d["team_appearances"], d["tournaments"], stages)

    results = evaluate(df)
    results.to_csv("reports/cv_results.csv", index=False)

    table = results.pivot_table(index=["task", "model"], columns="year", values="score")
    table["mean"] = table.mean(axis=1)
    pd.set_option("display.width", 140)
    print("\nGoals = RMSE (lower is better) | Stage = macro F1 (higher is better)\n")
    print(table.round(3))
