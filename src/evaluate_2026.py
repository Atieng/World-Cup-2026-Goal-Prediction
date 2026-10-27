"""Score submission_poisson_v2.csv against the real 2026 World Cup.

Run from the project root:  python3 -m src.evaluate_2026
Needs data/actual_2026.csv and submission-csv-files/submission_poisson_v2.csv.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, mean_squared_error

from src.features import load_data

ORDER = ["group", "roundof32", "roundof16", "qf", "sf", "runnerup", "champion"]

def main():
    _, _, _, _, _, test = load_data("data")
    sub = pd.read_csv("submission-csv-files/submission_poisson_v2.csv")
    actual = pd.read_csv("data/actual_2026.csv")

    # Attach country names if the submission only has IDs
    if "country" not in sub.columns:
        key = [c for c in sub.columns if c in test.columns][0]
        name = [c for c in test.columns if c.lower() in ("country", "team", "team_name", "country_name")][0]
        sub = sub.merge(test[[key, name]], on=key, how="left").rename(columns={name: "country"})
    goal_col = [c for c in sub.columns if "goal" in c.lower()][0]
    stage_col = [c for c in sub.columns if "stage" in c.lower()][0]
    print(f"Using columns: goals='{goal_col}', stage='{stage_col}'")

    df = sub.merge(actual, on="country", how="left")
    missing = df[df["actual_goals"].isna()]["country"].tolist()
    if missing:
        print("No actual result found for:", missing, "-> fix spelling in actual_2026.csv")
    df = df.dropna(subset=["actual_goals"])

    rmse = np.sqrt(mean_squared_error(df["actual_goals"], df[goal_col]))
    f1 = f1_score(df["actual_stage"], df[stage_col], average="macro", labels=ORDER)
    acc = (df["actual_stage"] == df[stage_col]).mean()
    base_rmse = np.sqrt(mean_squared_error(df["actual_goals"], np.full(len(df), df[goal_col].mean())))

    print(f"Teams scored     : {len(df)}")
    print(f"Goals RMSE       : {rmse:.3f}   (flat-mean baseline {base_rmse:.3f})")
    print(f"Stage macro F1   : {f1:.3f}")
    print(f"Stage exact acc. : {acc:.1%}")
    print(f"Predicted total  : {df[goal_col].sum():.0f} goals vs actual {df['actual_goals'].sum():.0f}")

    # Stage off-by: how many rounds each prediction missed by
    df["off_by"] = df[stage_col].map(ORDER.index) - df["actual_stage"].map(ORDER.index)
    print("\nRounds off (pred - actual):")
    print(df["off_by"].value_counts().sort_index().to_string())

    df["goal_err"] = df[goal_col] - df["actual_goals"]
    cols = ["country", goal_col, "actual_goals", "goal_err", stage_col, "actual_stage"]
    print("\nAll teams:")
    print(df.sort_values("actual_goals", ascending=False)[cols].to_string(index=False))
    df[cols].to_csv("reports/eval_2026.csv", index=False)
    print("\nSaved reports/eval_2026.csv")

if __name__ == "__main__":
    main()
