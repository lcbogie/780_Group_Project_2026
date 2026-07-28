This is the current, corrected round of the 2026 World Cup out-of-sample evaluation. It **supersedes** `Mohammed_2026_Evaluation_Data_DRAFT/`, which was built against an earlier version of Colby's modeling notebook, before he added Ordinal Rank as a feature and revised the notebook's conclusion. That folder is left as-is rather than overwritten -- see "What changed" below for why this round exists.


**For presenting:** `MODEL_DEMO.html` -- an interactive, in-browser version of every model this project has built, computed live in JS (nothing looked up from a server). Four tabs:
1. **Pick a real 2026 team** -- see all 9 scoring-margin models' predictions, lock in your own guess before revealing the real result, and see who (model or human) got closest.
2. **Build a hypothetical team** -- sliders for FIFA rank/points and roster composition; every model's prediction updates live.
3. **All 48 teams** -- a sortable leaderboard of every team's actual vs. predicted scoring margin, switchable across all 9 models, so you can show the whole out-of-sample result set at once instead of one team at a time.
4. **Predict a match** -- pick one of the 83 real, fair-comparison 2026 World Cup matches and see what Henmi's Poisson and XGBoost models actually predicted (win/draw/loss probabilities), guess the outcome yourself, then reveal what really happened.

## What changed since the DRAFT folder

Colby added **Ordinal Rank** (a team's literal FIFA rank position, e.g. 1st, 2nd, 3rd) to his modeling notebook as a feature distinct from raw **Ranking Points**, and found it's a meaningfully stronger predictor. His revised in-sample conclusion: `Ordinal Rank + t5league` is the best model, `Ordinal Rank only` beats `t5league only`, and the old "t5league beats FIFA rank" conclusion is explicitly superseded (he kept it in his notebook marked "flawed prior conclusion, for archival purposes" -- good practice, not deleted).

This folder redoes the out-of-sample 2026 evaluation against his **current** notebook, testing all 8 linear model variants (not the old 5) plus a rebuilt neural network, against real 2026 results.

## What's in this folder

**Data**
- `wc_2026_x_matrix.csv` and `wc_2026_model_input.csv` -- **corrected**: `fifa_rank` (ordinal position) and `fifa_ranking_points` (raw points) for all 48 teams now hold the official FIFA ranking dated 11 June 2026 (the real pre-tournament release), replacing the placeholder that had been pulled from an old 2024-06-20 snapshot. Everything else in these files (roster-by-league counts, real 2026 match results) is unchanged. 48 teams, 49 lines including header, in both files.
- `wc_2026_y_vector.csv` -- **unchanged** (carries no FIFA rank/points data, so there was nothing to correct here). Still real 2026 match results, 48 teams.
- `wc_2018.csv`, `wc_2022.csv`, `fifa_ranking-2024-06-20.csv` -- copies of the historical data used to train/sanity-check the models below, pulled fresh from Colby's current notebook folder. Not modified.
- `citations_EVAL_COPY.txt` -- a copy of the project's root `citations.txt` with sources added during this round appended at the bottom.

**Notebooks** (all working copies, pulled fresh from the teammates' current GitHub state -- their original folders were not modified)
- `ranking_points_and_t5league_modeling_EVAL_COPY.ipynb` -- copy of Colby's current `FIFA_Ranking_and_T5League_Modeling/ranking_points_and_t5league_modeling.ipynb`, extended to score all 8 of his linear regressions plus the neural network below against real 2026 results. **Best model: Ordinal Rank + t5league (aggregate), RMSE 0.931** -- clearly ahead of everything else tested.
- `ranking_points_and_t5league_modeling_NN_COPY.ipynb` -- same base notebook, with the still-stubbed "Approach 3: Neural Network" section filled in (scikit-learn `MLPRegressor`, using Ordinal Rank + individual leagues -- the strongest feature set per Colby's revised conclusion). In-sample RMSE 0.555 vs. 0.835 for the best linear model; out-of-sample RMSE 1.153, worse than 7 of the other 8 models. Overfitting, confirmed a second time (a different neural network overfit in the superseded DRAFT round too).
- `mohammed_baseline_models_2026_EVAL_COPY.ipynb` -- unaffected by Colby's revision (already used ordinal FIFA rank, not points), rebuilt here for consolidation. **Best model, both 2022 and 2026: Linear regression on FIFA rank.**

## Context: a complementary approach elsewhere on the team

Henmi built a separate pipeline (`Henmi_Player_Stats_XGBoost/`, not touched here) predicting match outcomes rather than tournament scoring margin, using ELO ratings + player performance stats. A Poisson baseline (68.7% accuracy) beat XGBoost (63.9%) on real 2026 matches -- the same "simpler model wins" pattern, via a completely different method. Worth citing in the final report as independent corroboration.

His Poisson and XGBoost models are also reproduced (not modified -- same architecture, same hyperparameters, same train/test split) in `MODEL_DEMO.html`'s "Predict a match" tab. His `PlayerStat_XG.ipynb` has a hardcoded local Windows path and needs a live API key to run end-to-end, so this reproduction instead starts from `final_match_feature_table.csv` (his own already-saved output) and re-runs just the modeling steps -- confirmed to reproduce his exact reported accuracy (68.7% / 63.9%) before anything was embedded in the demo.

## Still open

Our proposal's stated success criterion was to outperform Nate Silver's "Pele" system on the 2026 World Cup. That comparison is not yet possible: Silver's per-team numeric predictions are behind a paid Silver Bulletin subscription, and PELE's output (championship/advancement probability) isn't the same target variable as our `mean_scoring_margin`. See the "Attempting the proposal's actual benchmark" section of `ranking_points_and_t5league_modeling_EVAL_COPY.ipynb` for what we tried.
