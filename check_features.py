import pandas as pd
pd.set_option("display.max_columns", None)
from src.features import load_data, build_stages, prior_features, FEATURES

d = load_data("data")
stages = build_stages(d["Train"], d["tournament_standings"])
train_df = prior_features(d["team_appearances"], d["tournaments"], stages)

X, y_goals, y_stage = train_df[FEATURES], train_df["total_goals"], train_df["stage_numeric"]

print(train_df.shape)
print(train_df["stage_numeric"].value_counts())
brazil = train_df[train_df["team_id"] == "T-09"]
print(brazil[["year", "gf", "last_gf", "prev_apps", "prev_gf_pm"]].head(6))