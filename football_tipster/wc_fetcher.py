import json
from pathlib import Path

WC2026_DIR = Path(__file__).parent / "wc2026"
TEAMS_DIR = WC2026_DIR / "teams"
MATCHES_DIR = WC2026_DIR / "matches"


def _team_filename(team_name: str) -> str:
    return team_name.strip().lower().replace(" ", "_") + ".json"


def load_team_data(team_name: str) -> dict:
    path = TEAMS_DIR / _team_filename(team_name)
    if not path.exists():
        raise FileNotFoundError(
            f"Team data not found for '{team_name}' (expected at {path})"
        )
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_group_matches(group: str) -> list:
    path = MATCHES_DIR / f"group_{group.strip().lower()}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Group matches not found for group '{group}' (expected at {path})"
        )
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("matches", [])


def get_team_history(team_name: str) -> list:
    team = load_team_data(team_name)
    matches = list(team.get("recent_matches", []))
    matches.sort(key=lambda m: m.get("date", ""), reverse=True)
    return matches


def get_h2h(home_team: str, away_team: str) -> dict:
    try:
        team = load_team_data(home_team)
    except FileNotFoundError:
        return {}
    return team.get("h2h", {}).get(away_team, {})


def get_odds(match_id: str, group: str) -> dict | None:
    matches = load_group_matches(group)
    for m in matches:
        if m.get("match_id") == match_id:
            return {
                **m.get("odds", {}),
                "odds_confidence": m.get("odds_confidence"),
            }
    return None
