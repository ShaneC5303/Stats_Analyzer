"""
23DEA3202 - Data Exploration and Preparation
Project: Test Player Performance Analysis (Abstract Cricket)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ── Setup ──────────────────────────────────────────────────────────────────────
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("Set2")

# ── 1. Load Data ───────────────────────────────────────────────────────────────
def load_data():
    path = os.path.join(os.path.dirname(__file__), "..", "data", "cricket_players.csv")
    df = pd.read_csv(path)
    print("=" * 60)
    print("CRICKET PLAYER PERFORMANCE ANALYSIS")
    print("=" * 60)
    print(f"\n✅ Data loaded: {df.shape[0]} players, {df.shape[1]} features\n")
    return df

# ── 2. Data Overview ───────────────────────────────────────────────────────────
def data_overview(df):
    print("── Data Overview ──────────────────────────────────────────")
    print(df.dtypes.to_string())
    print(f"\nMissing values:\n{df.isnull().sum()}")
    print(f"\nBasic Stats:\n{df.describe().round(2)}")

# ── 3. Data Cleaning ───────────────────────────────────────────────────────────
def clean_data(df):
    print("\n── Data Cleaning ──────────────────────────────────────────")
    before = df.shape[0]
    df = df.drop_duplicates()
    df["years_active"] = 2024 - df["debut_year"]
    # Replace 0.0 bowling average with NaN for non-bowlers (cleaner analysis)
    df["bowling_avg"] = df["bowling_avg"].replace(0.0, np.nan)
    df["economy"]     = df["economy"].replace(0.0, np.nan)
    print(f"Rows before: {before} → after dedup: {df.shape[0]}")
    print(f"Added 'years_active' column.")
    return df

# ── 4. Exploratory Data Analysis ──────────────────────────────────────────────
def eda(df):
    print("\n── EDA ────────────────────────────────────────────────────")

    # Role distribution
    print("\nPlayers by Role:")
    print(df["role"].value_counts().to_string())

    # Top 5 run scorers
    print("\nTop 5 Run Scorers:")
    top_batsmen = df.nlargest(5, "runs")[["player_name","country","runs","average","centuries"]]
    print(top_batsmen.to_string(index=False))

    # Top 5 wicket takers
    print("\nTop 5 Wicket Takers:")
    top_bowlers = df.nlargest(5, "wickets")[["player_name","country","wickets","bowling_avg","economy"]]
    print(top_bowlers.to_string(index=False))

    # Country-wise average runs
    print("\nAverage Runs by Country:")
    print(df.groupby("country")["runs"].mean().round(1).sort_values(ascending=False).to_string())

# ── 5. Visualisations ─────────────────────────────────────────────────────────
def plot_charts(df):
    print("\n── Generating Charts ──────────────────────────────────────")
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Cricket Player Performance Analysis", fontsize=16, fontweight="bold")

    # 1. Role distribution (pie)
    role_counts = df["role"].value_counts()
    axes[0, 0].pie(role_counts, labels=role_counts.index, autopct="%1.1f%%", startangle=90)
    axes[0, 0].set_title("Player Role Distribution")

    # 2. Runs by country (bar)
    country_runs = df.groupby("country")["runs"].sum().sort_values(ascending=False)
    sns.barplot(x=country_runs.values, y=country_runs.index, ax=axes[0, 1], palette="Blues_r")
    axes[0, 1].set_title("Total Runs by Country")
    axes[0, 1].set_xlabel("Total Runs")

    # 3. Average vs Strike Rate (scatter)
    colors = {"Batsman": "blue", "Bowler": "red", "AllRounder": "green", "WicketKeeper": "orange"}
    for role, grp in df.groupby("role"):
        axes[0, 2].scatter(grp["average"], grp["strike_rate"],
                           label=role, color=colors.get(role, "gray"), s=80, alpha=0.8)
    axes[0, 2].set_xlabel("Batting Average")
    axes[0, 2].set_ylabel("Strike Rate")
    axes[0, 2].set_title("Average vs Strike Rate")
    axes[0, 2].legend(fontsize=8)

    # 4. Top 8 run scorers (bar)
    top8 = df.nlargest(8, "runs")
    sns.barplot(x="runs", y="player_name", data=top8, ax=axes[1, 0], palette="Greens_r")
    axes[1, 0].set_title("Top 8 Run Scorers")
    axes[1, 0].set_xlabel("Runs")
    axes[1, 0].set_ylabel("")

    # 5. Centuries vs Half-Centuries (bar grouped)
    top10 = df.nlargest(10, "runs")[["player_name", "centuries", "half_centuries"]].set_index("player_name")
    top10.plot(kind="bar", ax=axes[1, 1], colormap="Paired")
    axes[1, 1].set_title("Centuries & Half-Centuries (Top 10 Batsmen)")
    axes[1, 1].set_xlabel("")
    axes[1, 1].tick_params(axis="x", rotation=45)
    axes[1, 1].legend(fontsize=8)

    # 6. Correlation heatmap
    numeric_cols = ["runs", "average", "strike_rate", "centuries", "wickets", "catches", "years_active"]
    corr = df[numeric_cols].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=axes[1, 2], linewidths=0.5)
    axes[1, 2].set_title("Feature Correlation Heatmap")

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "cricket_analysis.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"✅ Chart saved → {out_path}")
    plt.show()

# ── 6. Performance Scoring ────────────────────────────────────────────────────
def performance_score(df):
    print("\n── Performance Scoring ────────────────────────────────────")
    d = df.copy()
    # Normalise key metrics 0-1
    for col in ["runs", "average", "strike_rate", "wickets", "catches"]:
        mn, mx = d[col].min(), d[col].max()
        d[f"{col}_norm"] = (d[col] - mn) / (mx - mn) if mx != mn else 0

    d["performance_score"] = (
        d["runs_norm"]       * 0.30 +
        d["average_norm"]    * 0.25 +
        d["strike_rate_norm"]* 0.15 +
        d["wickets_norm"]    * 0.20 +
        d["catches_norm"]    * 0.10
    ).round(4)

    ranked = d[["player_name", "country", "role", "performance_score"]]\
               .sort_values("performance_score", ascending=False)\
               .reset_index(drop=True)
    ranked.index += 1
    print(ranked.to_string())

    out_path = os.path.join(OUTPUT_DIR, "player_rankings.csv")
    ranked.to_csv(out_path, index_label="rank")
    print(f"\n✅ Rankings saved → {out_path}")
    return d

# ── 7. Summary ────────────────────────────────────────────────────────────────
def summary(df):
    print("\n── Summary Statistics ─────────────────────────────────────")
    print(f"  Total players analysed : {len(df)}")
    print(f"  Total runs (dataset)   : {df['runs'].sum():,}")
    print(f"  Highest score          : {df['highest_score'].max()} ({df.loc[df['highest_score'].idxmax(),'player_name']})")
    print(f"  Most wickets           : {df['wickets'].max()} ({df.loc[df['wickets'].idxmax(),'player_name']})")
    print(f"  Best batting average   : {df['average'].max():.2f} ({df.loc[df['average'].idxmax(),'player_name']})")
    print(f"  Best strike rate       : {df['strike_rate'].max():.2f} ({df.loc[df['strike_rate'].idxmax(),'player_name']})")
    print("\n✅ Analysis complete!")

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = load_data()
    data_overview(df)
    df = clean_data(df)
    eda(df)
    plot_charts(df)
    df = performance_score(df)
    summary(df)
