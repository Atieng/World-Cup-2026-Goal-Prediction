"""Score submission_poisson_v2.csv against the real 2026 World Cup.
Run from the project root:  python3 -m src.evaluate_2026
"""
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, mean_squared_error

ORDER = ["group", "roundof32", "roundof16", "qf", "sf", "runnerup", "champion"]

def main():
    test = pd.read_csv("data/Test.csv")
    sub = pd.read_csv("submission-csv-files/submission_poisson_v2.csv")
    actual = pd.read_csv("data/actual_2026.csv")

    # Submission rows are in the same order as Test.csv (same as predict.py)
    sub["country"] = test["country"].values
    df = sub.merge(actual, on="country", how="left")

    missing = df.loc[df["actual_goals"].isna(), "country"].tolist()
    if missing:
        print("No actual result for:", missing)
    df = df.dropna(subset=["actual_goals"])

    rmse = np.sqrt(mean_squared_error(df["actual_goals"], df["total_goals"]))
    base = np.sqrt(mean_squared_error(df["actual_goals"],
                                      np.full(len(df), df["total_goals"].mean())))
    f1 = f1_score(df["actual_stage"], df["Target"], average="macro", labels=ORDER)
    acc = (df["actual_stage"] == df["Target"]).mean()

    print(f"Teams scored     : {len(df)}")
    print(f"Goals RMSE       : {rmse:.3f}   (flat-mean baseline {base:.3f})")
    print(f"Goals correlation: {df['total_goals'].corr(df['actual_goals']):.2f}")
    print(f"Stage macro F1   : {f1:.3f}")
    print(f"Stage exact acc. : {acc:.1%}")
    print(f"Predicted total  : {df['total_goals'].sum():.0f} goals vs actual {df['actual_goals'].sum():.0f}")

    df["off_by"] = df["Target"].map(ORDER.index) - df["actual_stage"].map(ORDER.index)
    print("\nRounds off (predicted - actual):")
    print(df["off_by"].value_counts().sort_index().to_string())

    df["goal_err"] = (df["total_goals"] - df["actual_goals"]).round(2)
    cols = ["country", "total_goals", "actual_goals", "goal_err", "Target", "actual_stage"]
    print("\nAll teams:")
    print(df.sort_values("actual_goals", ascending=False)[cols].to_string(index=False))
    df[cols].to_csv("reports/eval_2026.csv", index=False)
    print("\nSaved reports/eval_2026.csv")

if __name__ == "__main__":
    main()
