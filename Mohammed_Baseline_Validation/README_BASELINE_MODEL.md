# Baseline Validation
This folder contains the initial baseline validation done prior to the midway report 

## Purpose
The purpose of this analysis is to establish simple benchmark models for predicting World Cup mean scoring margin.

The outcome variable is:
mean_scoring_margin = (goals_for - goals_against) / games_played

This baseline gives the team a reference point for comparing later player-feature, Poisson, Random Forest, and XGBoost models.

## Baseline setup

The preliminary validation setup trains on 2018 World Cup team-level outcomes and evaluates on 2022 World Cup team-level outcomes.

Models compared:
- Historical mean baseline
- Linear regression using FIFA rank
- Linear regression using FIFA ranking points
- Decision tree
- Random forest

Evaluation metrics:
- RMSE
- MAE

## Current recovered results
| Model | RMSE | MAE |
|---|---:|---:|
| Linear regression: FIFA rank | 0.9002 | 0.7212 |
| Decision tree | 0.9715 | 0.8280 |
| Random forest | 0.9767 | 0.7980 |
| Historical mean | 1.0224 | 0.8055 |
| Linear regression: FIFA points | 1.2481 | 1.0193 |

The best preliminary baseline was the FIFA-rank linear regression.

Recovered fitted model:

Y_hat = 0.2425 - 0.0181 * FIFA_rank
