import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="World Cup 2026 Predictions", page_icon="⚽", layout="wide")

STAGES = ["group", "roundof32", "roundof16", "qf", "sf", "runnerup", "champion"]
LABEL = {"group": "Group stage", "roundof32": "Round of 32", "roundof16": "Round of 16",
         "qf": "Quarter-final", "sf": "Semi-final", "runnerup": "Runner-up", "champion": "Champion"}
FEATURES = ["prev_apps", "first_appearance", "prev_gf_pm", "prev_ga_pm", "prev_win_rate",
            "last_gf", "last_stage", "best_stage", "years_since"]
NAME_FIXES = {"Czechia": "Czech Republic", "Turkiye": "Turkey",
              "Cote d'Ivoire": "Ivory Coast", "DR Congo": "Zaire"}
HIST_STAGE = {"group stage": 0, "second group stage": 3, "round of 16": 2, "quarter-finals": 3,
              "semi-finals": 4, "third-place match": 4, "final round": 4, "final": 5}


@st.cache_data
def load_results():
    ev = pd.read_csv("reports/eval_2026.csv")
    actual = pd.read_csv("data/actual_2026.csv")
    df = ev.merge(actual[["country", "group"]], on="country")
    df["rounds_off"] = df["Target"].map(STAGES.index) - df["actual_stage"].map(STAGES.index)
    return df


df = load_results()

st.title("⚽ Predicting the 2026 World Cup")
st.write(f"Loaded predictions for **{len(df)}** teams.")
with st.expander("Raw data"):
    st.dataframe(df, hide_index=True)

# ---------------- STEP 2: sidebar filters
st.sidebar.header("Filters")
groups = st.sidebar.multiselect("Groups", sorted(df["group"].unique()), default=sorted(df["group"].unique()))
stage_pick = st.sidebar.selectbox("Actual stage reached", ["All"] + [LABEL[s] for s in STAGES])
only_misses = st.sidebar.checkbox("Only show wrong stage predictions")

view = df[df["group"].isin(groups)]
if stage_pick != "All":
    view = view[view["actual_stage"].map(LABEL) == stage_pick]
if only_misses:
    view = view[view["rounds_off"] != 0]

rmse = np.sqrt(np.mean(view["goal_err"] ** 2)) if len(view) else 0
c1, c2, c3 = st.columns(3)
c1.metric("Teams shown", len(view))
c2.metric("Goals RMSE", f"{rmse:.2f}")
c3.metric("Exact stage", f"{(view['rounds_off'] == 0).mean():.0%}" if len(view) else "–")

# ---------------- STEP 3: interactive chart
st.subheader("Predicted vs actual goals")
chart = alt.Chart(view).mark_circle(size=120).encode(
    x=alt.X("total_goals:Q", title="Predicted goals"),
    y=alt.Y("actual_goals:Q", title="Actual goals"),
    color=alt.Color("actual_stage:N", sort=STAGES, scale=alt.Scale(domain=STAGES, scheme="viridis"), title="Actual stage"),
    tooltip=["country", "group", "total_goals", "actual_goals", "Target", "actual_stage"],
).interactive()
line = alt.Chart(pd.DataFrame({"x": [0, 21], "y": [0, 21]})).mark_line(
    strokeDash=[4, 4], color="grey").encode(x="x", y="y")
st.altair_chart(chart + line, use_container_width=True)
st.caption("Dashed line = perfect prediction. Points above it scored more than predicted. Scroll to zoom, drag to pan.")

# ---------------- STEP 4: team explorer
@st.cache_data
def load_history():
    ta = pd.read_csv("data/team_appearances.csv")
    ta = ta[ta["tournament_name"].str.contains("Men's")]
    ta = ta.assign(year=ta["tournament_name"].str[:4].astype(int), s=ta["stage_name"].map(HIST_STAGE))
    h = ta.groupby(["team_name", "year"]).agg(
        gf=("goals_for", "sum"), ga=("goals_against", "sum"),
        mp=("match_id", "nunique"), w=("win", "sum"), stage=("s", "max")).reset_index()
    stn = pd.read_csv("data/tournament_standings.csv")
    stn = stn[stn["tournament_name"].str.contains("Men's")]
    stn = stn.assign(year=stn["tournament_name"].str[:4].astype(int))
    pos = stn.set_index(["team_name", "year"])["position"].to_dict()
    p = np.array([pos.get(k, 0) for k in zip(h["team_name"], h["year"])])
    h.loc[p == 1, "stage"] = 6
    h.loc[p == 2, "stage"] = 5
    h.loc[(h["stage"] == 5) & (p != 1) & (p != 2), "stage"] = 4
    return h


hist = load_history()

st.divider()
st.header("🔎 Team explorer")
team = st.selectbox("Pick a team", sorted(df["country"]), index=sorted(df["country"]).index("Spain"))
row = df[df["country"] == team].iloc[0]
h = hist[hist["team_name"] == NAME_FIXES.get(team, team)]

a, b, c = st.columns(3)
a.metric("Predicted goals", f"{row['total_goals']:.1f}", f"actual {row['actual_goals']}", delta_color="off")
b.metric("Predicted stage", LABEL[row["Target"]])
c.metric("Actual stage", LABEL[row["actual_stage"]])

if h.empty:
    st.info(f"{team} made its World Cup debut in 2026, so the model had no history to use.")
else:
    st.bar_chart(h.set_index("year")["gf"], y_label="Goals scored", x_label="World Cup")

# ---------------- STEP 5: what-if predictor
@st.cache_resource
def load_model():
    return joblib.load("models/poisson_goals.joblib")


def team_features(h):
    """The 9 model inputs, computed from a team's World Cups before 2026."""
    if h.empty:
        return dict(prev_apps=0, first_appearance=1, prev_gf_pm=0.0, prev_ga_pm=0.0, prev_win_rate=0.0,
                    last_gf=0, last_stage=-1, best_stage=-1, years_since=99)
    h = h.sort_values("year")
    return dict(prev_apps=len(h), first_appearance=0,
                prev_gf_pm=h["gf"].sum() / h["mp"].sum(), prev_ga_pm=h["ga"].sum() / h["mp"].sum(),
                prev_win_rate=h["w"].sum() / h["mp"].sum(), last_gf=int(h["gf"].iloc[-1]),
                last_stage=int(h["stage"].iloc[-1]), best_stage=int(h["stage"].max()),
                years_since=2026 - int(h["year"].iloc[-1]))


SLOTS = {"champion": 1, "runnerup": 1, "sf": 2, "qf": 4, "roundof16": 8, "roundof32": 16, "group": 16}


def stage_from_rank(rank):
    """Rank 1 = strongest team. Fill the fixed 2026 places from the top."""
    for s in STAGES[::-1]:
        if rank <= SLOTS[s]:
            return s
        rank -= SLOTS[s]


model = load_model()
st.divider()
st.header(f"🎛️ What if? Change {team}'s history")
st.write("Move the sliders to see how the model's forecast would change. "
         "Each team keeps its own slider positions.")

f = team_features(h)
stage_names = {-1: "Never played", **{i: LABEL[s] for i, s in enumerate(STAGES)}}
k = team  # a key per team, so switching team gives that team's own sliders
left, right = st.columns(2)
with left:
    f["prev_gf_pm"] = st.slider("Goals scored per match (all past World Cups)", 0.0, 4.0, float(round(f["prev_gf_pm"], 2)), 0.05, key=k + "gf")
    f["prev_ga_pm"] = st.slider("Goals conceded per match", 0.0, 4.0, float(round(f["prev_ga_pm"], 2)), 0.05, key=k + "ga")
    f["prev_win_rate"] = st.slider("Win rate", 0.0, 1.0, float(round(f["prev_win_rate"], 2)), 0.01, key=k + "wr")
    f["last_gf"] = st.slider("Goals at the last World Cup", 0, 20, int(f["last_gf"]), key=k + "lg")
with right:
    f["prev_apps"] = st.slider("Previous World Cups played", 0, 22, int(f["prev_apps"]), key=k + "apps")
    f["first_appearance"] = int(f["prev_apps"] == 0)
    f["last_stage"] = st.select_slider("Stage reached last time", list(stage_names), f["last_stage"], format_func=stage_names.get, key=k + "ls")
    f["best_stage"] = st.select_slider("Best stage ever", list(stage_names), f["best_stage"], format_func=stage_names.get, key=k + "bs")
    f["years_since"] = st.slider("Years since last World Cup", 4, 99, int(f["years_since"]) if f["years_since"] >= 4 else 4, key=k + "ys")

new_goals = float(model.predict(pd.DataFrame([f])[FEATURES])[0])
others = df.loc[df["country"] != team, "total_goals"]
rank = int((others > new_goals).sum()) + 1
new_stage = stage_from_rank(rank)

x, y, z = st.columns(3)
x.metric("What-if goals", f"{new_goals:.1f}", f"{new_goals - row['total_goals']:+.1f} vs real forecast")
y.metric("Rank among 48", f"#{rank}")
z.metric("What-if stage", LABEL[new_stage])

# ---------------- STEP 6: head-to-head
st.divider()
st.header("⚔️ Compare two teams")
p, q = st.columns(2)
t1 = p.selectbox("Team A", sorted(df["country"]), index=sorted(df["country"]).index("Argentina"))
t2 = q.selectbox("Team B", sorted(df["country"]), index=sorted(df["country"]).index("Brazil"))

both = hist[hist["team_name"].isin([NAME_FIXES.get(t1, t1), NAME_FIXES.get(t2, t2)])]
if both.empty:
    st.info("Neither team has World Cup history before 2026.")
else:
    st.altair_chart(alt.Chart(both).mark_line(point=True).encode(
        x=alt.X("year:O", title=None), y=alt.Y("gf:Q", title="Goals scored"),
        color=alt.Color("team_name:N", title=None),
        tooltip=["team_name", "year", "gf", "mp"]), use_container_width=True)

cmp_ = df[df["country"].isin([t1, t2])].set_index("country")[["total_goals", "actual_goals", "Target", "actual_stage"]]
st.dataframe(cmp_.rename(columns={"total_goals": "Predicted goals", "actual_goals": "Actual goals",
                                  "Target": "Predicted stage", "actual_stage": "Actual stage"}))
