"""
Mohammed Baseline Models — World Cup Mean Scoring Margin
Reconstructed from the DATA 780 project workflow.

Update paths and column mappings before running.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

DATA_DIR = Path("../data")
FIGURE_DIR = Path("../figures")
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def clean_team_name(name):
    if pd.isna(name):
        return name
    name = str(name).strip().upper()
    replacements = {
        "UNITED STATES": "USA",
        "UNITED STATES OF AMERICA": "USA",
        "IRAN": "IR IRAN",
        "KOREA REPUBLIC": "SOUTH KOREA",
        "KOREA, REPUBLIC OF": "SOUTH KOREA",
    }
    return replacements.get(name, name)


def compute_team_tournament_summary(team_match_df, team_col="team", gf_col="goals_for", ga_col="goals_against"):
    df = team_match_df.copy()
    df["team_clean"] = df[team_col].map(clean_team_name)
    out = (
        df.groupby("team_clean", as_index=False)
          .agg(goals_for=(gf_col, "sum"), goals_against=(ga_col, "sum"), games_played=(team_col, "count"))
    )
    out["mean_scoring_margin"] = (out["goals_for"] - out["goals_against"]) / out["games_played"]
    return out


def match_rows_to_team_rows(match_df, team1_col, team2_col, goals1_col, goals2_col):
    a = match_df[[team1_col, team2_col, goals1_col, goals2_col]].copy()
    a.columns = ["team", "opponent", "goals_for", "goals_against"]
    b = match_df[[team2_col, team1_col, goals2_col, goals1_col]].copy()
    b.columns = ["team", "opponent", "goals_for", "goals_against"]
    return pd.concat([a, b], ignore_index=True)


def evaluate_regression(y_true, y_pred):
    return {"RMSE": mean_squared_error(y_true, y_pred, squared=False), "MAE": mean_absolute_error(y_true, y_pred)}


def run_baseline_models(train, test, random_state=42):
    y_train = train["mean_scoring_margin"].values
    y_test = test["mean_scoring_margin"].values
    results = []
    predictions = pd.DataFrame({"team_clean": test["team_clean"], "actual": y_test})

    hist_pred = np.repeat(y_train.mean(), len(test))
    predictions["pred_historical_mean"] = hist_pred
    results.append({"Model": "Historical mean", **evaluate_regression(y_test, hist_pred)})

    rank_model = LinearRegression().fit(train[["fifa_rank"]], y_train)
    rank_pred = rank_model.predict(test[["fifa_rank"]])
    predictions["pred_fifa_rank_lr"] = rank_pred
    results.append({"Model": "Linear regression: FIFA rank", **evaluate_regression(y_test, rank_pred)})

    points_model = LinearRegression().fit(train[["fifa_points"]], y_train)
    points_pred = points_model.predict(test[["fifa_points"]])
    predictions["pred_fifa_points_lr"] = points_pred
    results.append({"Model": "Linear regression: FIFA points", **evaluate_regression(y_test, points_pred)})

    tree = DecisionTreeRegressor(max_depth=3, min_samples_leaf=3, random_state=random_state)
    tree.fit(train[["fifa_rank", "fifa_points"]], y_train)
    tree_pred = tree.predict(test[["fifa_rank", "fifa_points"]])
    predictions["pred_decision_tree"] = tree_pred
    results.append({"Model": "Decision tree", **evaluate_regression(y_test, tree_pred)})

    forest = RandomForestRegressor(n_estimators=500, max_depth=3, min_samples_leaf=3, random_state=random_state)
    forest.fit(train[["fifa_rank", "fifa_points"]], y_train)
    forest_pred = forest.predict(test[["fifa_rank", "fifa_points"]])
    predictions["pred_random_forest"] = forest_pred
    results.append({"Model": "Random forest", **evaluate_regression(y_test, forest_pred)})

    return pd.DataFrame(results).sort_values("RMSE"), predictions, {"rank_model": rank_model, "points_model": points_model, "tree": tree, "forest": forest}


def plot_actual_vs_predicted(predictions, pred_col="pred_fifa_rank_lr", output_path=FIGURE_DIR / "actual_vs_predicted_2022.png"):
    df = predictions.copy()
    df["abs_error"] = (df["actual"] - df[pred_col]).abs()
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(df[pred_col], df["actual"], alpha=0.8)
    min_val = min(df[pred_col].min(), df["actual"].min())
    max_val = max(df[pred_col].max(), df["actual"].max())
    ax.plot([min_val, max_val], [min_val, max_val], linestyle="--")
    for _, row in df.nlargest(5, "abs_error").iterrows():
        ax.annotate(row["team_clean"], (row[pred_col], row["actual"]), xytext=(5, 5), textcoords="offset points", fontsize=8)
    ax.set_xlabel("Predicted mean scoring margin")
    ax.set_ylabel("Actual mean scoring margin")
    ax.set_title("Actual vs Predicted 2022 Mean Scoring Margin")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    return fig, ax


if __name__ == "__main__":
    print("This is a reusable baseline script. Import the functions or add your data-loading code here.")
    print("Reported earlier result: FIFA-rank LR RMSE=0.9002, MAE=0.7212; Y_hat=0.2425-0.0181R")
