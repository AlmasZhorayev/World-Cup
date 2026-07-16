import argparse
import warnings
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    confusion_matrix, classification_report, log_loss
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore")

RESULTS_PATH   = "results.csv"
SHOOTOUTS_PATH = "shootouts.csv"

# Knockout stage date windows for major tournaments
KNOCKOUT_WINDOWS = [
    ("UEFA Euro",     "2012-06-21", "2012-07-02"),
    ("UEFA Euro",     "2016-06-25", "2016-07-11"),
    ("UEFA Euro",     "2021-06-11", "2021-07-11"),
    ("UEFA Euro",     "2024-06-29", "2024-07-14"),
    ("FIFA World Cup","2014-06-28", "2014-07-13"),
    ("FIFA World Cup","2018-06-30", "2018-07-15"),
    ("FIFA World Cup","2022-12-03", "2022-12-18"),
    ("FIFA World Cup","2026-06-29", "2026-07-19"),
    ("Copa America",  "2015-06-17", "2015-07-04"),
    ("Copa America",  "2016-06-09", "2016-06-26"),
    ("Copa America",  "2019-07-02", "2019-07-07"),
    ("Copa America",  "2021-07-02", "2021-07-10"),
    ("Copa America",  "2024-07-05", "2024-07-14"),
    ("AFC Asian Cup", "2023-02-01", "2024-02-10"),
    ("Africa Cup of Nations", "2024-02-02", "2024-02-11"),
]

# Tournaments to exclude (non-competitive)
EXCLUDED_TOURNAMENTS = {"Friendly", "Friendly tournament"}

# Recency cutoff
RECENCY_CUTOFF = pd.Timestamp("2010-01-01")

# Current World Cup flag
CURRENT_WC_YEAR = 2026
CURRENT_WC_TOURNAMENT = "FIFA World Cup"

def load_data(results_path: str, shootouts_path: str) -> pd.DataFrame:
    """Load and merge results + shootouts; resolve winner for all rows."""
    results   = pd.read_csv(results_path)
    shootouts = pd.read_csv(shootouts_path)

    results["date"]   = pd.to_datetime(results["date"])
    shootouts["date"] = pd.to_datetime(shootouts["date"])

    # Drop any pre-existing winner columns to avoid conflicts
    results = results.drop(
        columns=[c for c in results.columns
                 if c in {"winner", "first_shooter", "winner_result",
                           "first_shooter_result", "winner_shootout",
                           "first_shooter_shootout"}],
        errors="ignore"
    )

    # Merge shootout winners
    results = results.merge(
        shootouts[["date", "home_team", "away_team", "winner"]],
        on=["date", "home_team", "away_team"],
        how="left"
    )

    # Fill winner from score where not a shootout
    results.loc[
        results["winner"].isna() & (results["home_score"] > results["away_score"]),
        "winner"
    ] = results["home_team"]
    results.loc[
        results["winner"].isna() & (results["home_score"] < results["away_score"]),
        "winner"
    ] = results["away_team"]

    # Remaining NaN = draw
    results["winner"] = results["winner"].fillna("Draw")

    return results


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add is_knockout and current_WC feature columns."""
    df = df.copy()

    # Knockout stage flag
    df["is_knockout"] = 0
    tournament_norm = df["tournament"].astype(str).str.strip().str.casefold()
    for tournament, start, end in KNOCKOUT_WINDOWS:
        mask = (
            tournament_norm.eq(tournament.casefold())
            & df["date"].between(pd.Timestamp(start), pd.Timestamp(end), inclusive="both")
        )
        df.loc[mask, "is_knockout"] = 1

    # Current World Cup flag (gives the model signal about WC 2026 form)
    df["current_WC"] = (
        df["tournament"].astype(str).str.strip().str.casefold()
            .eq(CURRENT_WC_TOURNAMENT.casefold())
        & df["date"].dt.year.eq(CURRENT_WC_YEAR)
    ).astype(int)

    return df


def build_team_perspective(df: pd.DataFrame, focus_teams: set) -> pd.DataFrame:
    """
    Expand the dataset so each match involving a focus team
    produces one row FROM THAT TEAM'S PERSPECTIVE.

    This lets the model learn team-specific win rates without
    separate models per team.
    """
    rows = []
    for _, match in df.iterrows():
        for team in focus_teams:
            if match["home_team"] == team:
                opponent = match["away_team"]
                is_home  = 1
            elif match["away_team"] == team:
                opponent = match["home_team"]
                is_home  = 0
            else:
                continue

            rows.append({
                "date":       match["date"],
                "team":       team,
                "opponent":   opponent,
                "tournament": match["tournament"],
                "neutral":    int(str(match["neutral"]).strip().upper() == "TRUE"),
                "is_home":    is_home,
                "is_knockout":match["is_knockout"],
                "current_WC": match["current_WC"],
                "winner":     match["winner"],
                # target: 1 = team won, 0 = did not win (loss or draw handled separately)
                "target":     int(match["winner"] == team),
                "is_draw":    int(match["winner"] == "Draw"),
            })

    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


# Model Training

def build_pipeline() -> Pipeline:
    categorical = ["team", "opponent", "tournament"]
    numeric     = ["neutral", "is_home", "is_knockout", "current_WC"]

    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
        ("num", "passthrough", numeric),
    ])

    model = CatBoostClassifier(
        iterations=300,
        depth=4,
        learning_rate=0.05,
        l2_leaf_reg=8,
        loss_function="Logloss",
        verbose=False,
        random_state=42,
        auto_class_weights="Balanced",
        allow_writing_files=False,
    )

    return Pipeline([("preprocess", preprocessor), ("model", model)])


def train(team_df: pd.DataFrame, verbose: bool = True):
    """
    Train two binary models:
      - win_pipeline:  P(team wins)
      - draw_pipeline: P(match is a draw)  [used only for group-stage context]

    Returns fitted pipelines and evaluation info.
    """
    FEATURES = ["team", "opponent", "tournament", "neutral",
                "is_home", "is_knockout", "current_WC"]

    # Win model
    win_df = team_df[team_df["is_draw"] == 0].copy()   # exclude draws for win/loss model
    X_win  = win_df[FEATURES]
    y_win  = win_df["target"]

    # Chronological split: test set must have ≥5 losses and ≥15 wins
    test_n = next(
        (n for n in range(15, len(y_win))
         if (y_win.iloc[-n:] == 0).sum() >= 5 and (y_win.iloc[-n:] == 1).sum() >= 15),
        None
    )
    if test_n is None or len(y_win) - test_n < 20:
        raise ValueError(
            "Not enough data for a meaningful chronological split. "
            "Check that the teams have enough historical matches."
        )
    split = len(X_win) - test_n
    X_train, X_test = X_win.iloc[:split], X_win.iloc[split:]
    y_train, y_test = y_win.iloc[:split], y_win.iloc[split:]

    win_pipeline = build_pipeline()
    win_pipeline.fit(X_train, y_train)

    if verbose:
        test_preds = win_pipeline.predict(X_test).astype(int)
        test_proba = win_pipeline.predict_proba(X_test)[:, 1]
        baseline   = y_test.value_counts(normalize=True).max()
        print("\n── Win/Loss Model Evaluation ──────────────────────────")
        print(f"  Training rows : {len(X_train)}")
        print(f"  Test rows     : {len(X_test)}")
        print(f"  Naive baseline: {baseline:.2%}")
        print(f"  Test accuracy : {accuracy_score(y_test, test_preds):.2%}")
        print(f"  Balanced acc  : {balanced_accuracy_score(y_test, test_preds):.2%}")
        print(f"  Log loss      : {log_loss(y_test, test_proba):.4f}")
        print(classification_report(y_test, test_preds,
              target_names=["Loss", "Win"], zero_division=0))

    # ---- Draw model ----
    X_all  = team_df[FEATURES]
    y_draw = team_df["is_draw"]

    draw_pipeline = build_pipeline()
    draw_pipeline.fit(X_all, y_draw)

    return win_pipeline, draw_pipeline


# Prediction

def predict_match(team1: str, team2: str, win_pipeline, 
                  draw_pipeline, tournament: str  = "FIFA World Cup", 
                  is_knockout: int = 1, neutral: int = 1, 
                  current_WC: int = 1, verbose: bool = True) -> dict:
    """
    Predict win/draw/loss probabilities for team1 vs team2.

    The model is run TWICE (once from each team's perspective) and the
    results are averaged for consistency.

    Returns a dict with keys: team1_win, draw, team2_win
    """

    def make_row(team, opponent, home):
        return pd.DataFrame([{
            "team":       team,
            "opponent":   opponent,
            "tournament": tournament,
            "neutral":    neutral,
            "is_home":    home,
            "is_knockout":is_knockout,
            "current_WC": current_WC,
        }])

    # View from team1's perspective (team1 as "home" in neutral = doesn't matter much)
    row1 = make_row(team1, team2, 1)
    row2 = make_row(team2, team1, 0)   # team2's perspective (team1 is opponent)

    # Win probabilities (excluding draw)
    p_team1_wins_v1 = win_pipeline.predict_proba(row1)[0][1]
    p_team2_wins_v2 = win_pipeline.predict_proba(row2)[0][1]

    # Average for symmetry
    p_team1_win_raw = (p_team1_wins_v1 + (1 - p_team2_wins_v2)) / 2

    # Draw probability (average of both perspectives)
    p_draw_v1 = draw_pipeline.predict_proba(row1)[0][1]
    p_draw_v2 = draw_pipeline.predict_proba(row2)[0][1]
    p_draw_raw = (p_draw_v1 + p_draw_v2) / 2

    # Combine: P(win1) + P(draw) + P(win2) must sum to 1
    # Scale win probability to the non-draw share
    non_draw = max(1 - p_draw_raw, 1e-9)
    p_team1_win  = p_team1_win_raw * non_draw
    p_team2_win  = (1 - p_team1_win_raw) * non_draw
    p_draw       = p_draw_raw

    # Renormalize to exactly 1.0
    total = p_team1_win + p_draw + p_team2_win
    p_team1_win /= total
    p_draw      /= total
    p_team2_win /= total

    if verbose:
        print(f"\nMatch Prediction")
        print(f"  {team1:<20} vs  {team2}")
        print(f"  Tournament  : {tournament}")
        print(f"  Stage       : {'Knockout' if is_knockout else 'Group stage'}")
        print(f"  Venue       : {'Neutral' if neutral else 'Home/Away'}")
        print(f"  ─────────────────────────────────────────────────────")
        print(f"  {team1} win  : {p_team1_win:.1%}")
        print(f"  Draw        : {p_draw:.1%}")
        print(f"  {team2} win  : {p_team2_win:.1%}")
        print(f"  ─────────────────────────────────────────────────────")
        if p_team1_win > p_team2_win:
            margin = p_team1_win - p_team2_win
            print(f"  Favoured    : {team1}  (+{margin:.1%})")
        elif p_team2_win > p_team1_win:
            margin = p_team2_win - p_team1_win
            print(f"  Favoured    : {team2}  (+{margin:.1%})")
        else:
            print(f"  Prediction  : Even matchup")

    return {
        f"{team1}_win": p_team1_win,
        "draw":          p_draw,
        f"{team2}_win":  p_team2_win,
    }


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Predict international football match outcomes.")
    parser.add_argument("--team1",      default="France",          help="First team name")
    parser.add_argument("--team2",      default="Spain",           help="Second team name")
    parser.add_argument("--tournament", default="FIFA World Cup",  help="Tournament name")
    parser.add_argument("--knockout",   action="store_true", default=True, help="Knockout stage?")
    parser.add_argument("--neutral",    action="store_true", default=True, help="Neutral venue?")
    parser.add_argument("--current-wc", action="store_true", default=True, help="Is this the 2026 WC?")
    args = parser.parse_args()

    print("Loading data...")
    results = load_data(RESULTS_PATH, SHOOTOUTS_PATH)

    print("Preprocessing...")
    results = add_features(results)

    # Filter out friendlies and very old matches
    results = results[
        ~results["tournament"].isin(EXCLUDED_TOURNAMENTS)
        & (results["date"] >= RECENCY_CUTOFF)
    ]

    # We train the model on ALL teams that appear in the data
    all_teams = set(results["home_team"]) | set(results["away_team"])
    print(f"Building team-perspective dataset ({len(all_teams)} teams)...")
    team_df = build_team_perspective(results, all_teams)

    print("Training model...")
    win_pipeline, draw_pipeline = train(team_df, verbose=True)

    predict_match(
        team1       = args.team1,
        team2       = args.team2,
        win_pipeline  = win_pipeline,
        draw_pipeline = draw_pipeline,
        tournament  = args.tournament,
        is_knockout = int(args.knockout),
        neutral     = int(args.neutral),
        current_WC  = int(args.current_wc),
        verbose     = True,
    )


if __name__ == "__main__":
    main()