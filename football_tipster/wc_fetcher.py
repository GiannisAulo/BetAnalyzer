import json
from pathlib import Path

WC2026_DIR = Path(__file__).parent / "wc2026"
TEAMS_DIR = WC2026_DIR / "teams"
MATCHES_DIR = WC2026_DIR / "matches"

_API_ID_INDEX: dict[int, Path] | None = None


def _build_api_id_index() -> dict[int, Path]:
    index = {}
    for path in TEAMS_DIR.glob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            api_id = data.get("api_id")
            if api_id is not None:
                index[int(api_id)] = path
        except Exception:
            pass
    return index


def get_api_id_index() -> dict[int, Path]:
    global _API_ID_INDEX
    if _API_ID_INDEX is None:
        _API_ID_INDEX = _build_api_id_index()
    return _API_ID_INDEX


def load_team_by_api_id(api_id: int) -> dict:
    index = get_api_id_index()
    path = index.get(api_id)
    if path is None:
        raise FileNotFoundError(f"No team file found for api_id={api_id}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_team_path_by_api_id(api_id: int) -> Path:
    index = get_api_id_index()
    path = index.get(api_id)
    if path is None:
        raise FileNotFoundError(f"No team file found for api_id={api_id}")
    return path


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
