"""
Filter international match results (results.csv) down to:
  1. The 2018-2026 date window
  2. Matches involving at least one of the 48 qualified 2026 World Cup teams
  3. Adds a competition-tier weight column for the Poisson model

Usage:
    python3 filter_results.py results.csv
Output:
    filtered_results_2018_2026.csv
"""

import pandas as pd

INPUT_PATH = "results.csv"
OUTPUT_PATH = "filtered_results_2018_2026.csv"

START_DATE = "2018-01-01"

# The 48 qualified 2026 World Cup teams, spelled to match this dataset's
# naming convention (confirmed against results.csv team names directly --
# note several differ from FIFA's official naming or common media naming).
QUALIFIED_TEAMS_2026 = {
    # Host nations + Round of 16 (as of July 2026)
    "Canada", "Brazil", "Paraguay", "Morocco", "Norway", "France", "Mexico",
    "England", "Belgium", "United States", "Spain", "Portugal", "Switzerland",
    "Egypt", "Argentina", "Colombia",
    # Round of 32 (eliminated)
    "South Africa", "Japan", "Germany", "Netherlands", "Ivory Coast", "Sweden",
    "Ecuador", "DR Congo", "Senegal", "Bosnia and Herzegovina", "Austria",
    "Croatia", "Algeria", "Australia", "Cape Verde", "Ghana",
    # Group stage (eliminated)
    "Czech Republic", "South Korea", "Qatar", "Scotland", "Haiti", "Turkey",
    "Curaçao", "Tunisia", "New Zealand", "Iran", "Saudi Arabia", "Uruguay",
    "Iraq", "Jordan", "Uzbekistan", "Panama",
}

assert len(QUALIFIED_TEAMS_2026) == 48, f"Expected 48 teams, got {len(QUALIFIED_TEAMS_2026)}"

# Competition-tier weights for the Poisson model, loosely modeled on
# FIFA's own match-importance weighting in its ranking methodology.
# Anything not listed here defaults to 1.0 (see get_tier_weight below).
TIER_WEIGHTS = {
    "FIFA World Cup": 3.0,
    "FIFA World Cup qualification": 2.0,
    "UEFA Euro qualification": 1.5,
    "UEFA Euro": 2.5,
    "UEFA Nations League": 1.5,
    "Copa América": 2.0,
    "African Cup of Nations": 2.0,
    "African Cup of Nations qualification": 1.5,
    "AFC Asian Cup": 2.0,
    "AFC Asian Cup qualification": 1.5,
    "CONCACAF Nations League": 1.5,
    "Gold Cup": 1.5,
    "Friendly": 0.5,
}
DEFAULT_TIER_WEIGHT = 1.0


def get_tier_weight(tournament: str) -> float:
    return TIER_WEIGHTS.get(tournament, DEFAULT_TIER_WEIGHT)


def filter_results(input_path: str = INPUT_PATH) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    df["date"] = pd.to_datetime(df["date"])

    # 1. Date window
    df = df[df["date"] >= START_DATE].copy()

    # 2. Drop unplayed/future fixtures (no score yet)
    df = df[df["home_score"].notna() & df["away_score"].notna()].copy()
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)

    # 3. Keep matches involving at least one qualified team
    #    (use "at least one" rather than "both" so you retain matches
    #    against non-qualified opponents -- still useful signal for a
    #    qualified team's attack/defense rate estimate)
    mask = (
        df["home_team"].isin(QUALIFIED_TEAMS_2026)
        | df["away_team"].isin(QUALIFIED_TEAMS_2026)
    )
    df = df[mask].copy()

    # 4. Add tier weight column for the weighted Poisson fit
    df["tier_weight"] = df["tournament"].apply(get_tier_weight)

    # 5. Flag whether each side is a qualified 2026 team (useful for
    #    later filtering to "both qualified" matches for evaluation)
    df["home_is_qualified"] = df["home_team"].isin(QUALIFIED_TEAMS_2026)
    df["away_is_qualified"] = df["away_team"].isin(QUALIFIED_TEAMS_2026)

    return df.reset_index(drop=True)


if __name__ == "__main__":
    import sys
    input_path = sys.argv[1] if len(sys.argv) > 1 else INPUT_PATH
    filtered = filter_results(input_path)
    filtered.to_csv(OUTPUT_PATH, index=False)

    print(f"Filtered dataset: {len(filtered)} matches -> {OUTPUT_PATH}")
    print(f"Date range: {filtered['date'].min().date()} to {filtered['date'].max().date()}")
    print(f"Matches with BOTH teams qualified: {(filtered['home_is_qualified'] & filtered['away_is_qualified']).sum()}")
    print(f"Matches with exactly one team qualified: {(filtered['home_is_qualified'] ^ filtered['away_is_qualified']).sum()}")
    print()
    print("Tier weight distribution:")
    print(filtered["tier_weight"].value_counts().sort_index(ascending=False))
