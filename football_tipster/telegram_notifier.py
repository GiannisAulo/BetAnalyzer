import os
import requests

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

def send_telegram_message(text: str) -> None:
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("[Telegram] No token/chat_id found, skipping.")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": "HTML"
    }
    try:
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        print("[Telegram] Message sent.")
    except Exception as e:
        print(f"[Telegram] Failed to send message: {e}")

def build_telegram_message(value_picks: list, accas: list) -> str:
    lines = ["🎯 <b>BetAnalyzer Daily Picks</b>\n"]

    if not value_picks:
        lines.append("No value picks found today.")
    else:
        for pick in value_picks:
            home = pick.get("home", "")
            away = pick.get("away", "")
            market = pick.get("market", "")
            selection = pick.get("pick", "")
            prob = pick.get("model_prob", 0)
            # edge is stored in percent units (e.g. 6.7 == +6.7%), not a 0–1 fraction
            edge = pick.get("edge") or 0
            # stake_units is None when Kelly recommends sub-1u (don't bet); show "—"
            stake = pick.get("stake_units")
            stake_str = f"{stake}u" if stake is not None else "—"
            lines.append(
                f"⚽ <b>{home} vs {away}</b>\n"
                f"   {market} → {selection}\n"
                f"   Prob: {prob:.0%} | Edge: +{edge:.1f}% | Stake: {stake_str}\n"
            )

    if accas:
        lines.append("\n📦 <b>Accumulators:</b>")
        for idx, acca in enumerate(accas, 1):
            # acca dict uses "joint_odds" (set by markets.build_cross_fixture_accas)
            odds = acca.get("joint_odds", 0)
            prob = acca.get("joint_prob", 0)
            stake = acca.get("stake_units")
            stake_str = f"{stake}u" if stake is not None else "—"
            # VERIFIED = every leg has real bookmaker odds; MODEL = at least one
            # leg uses model-derived fair odds (user should verify real price).
            tag = "Verified" if acca.get("verified_edge") else "Model"
            lines.append(
                f"\n<b>Acca {idx}</b> ({tag}) | "
                f"Combined: {odds:.2f} | Prob: {prob:.0%} | Stake: {stake_str}"
            )
            for leg in acca.get("legs", []):
                home   = leg.get("home", "")
                away   = leg.get("away", "")
                pick   = leg.get("pick", "")
                leg_o  = leg.get("odds")
                # ~ suffix marks an odds value derived from the model, not the bookmaker
                mark   = "~" if leg.get("inferred_odds") else ""
                odds_part = f" @{leg_o:.2f}{mark}" if leg_o is not None else ""
                lines.append(f"   • {home} vs {away} — {pick}{odds_part}")

    lines.append(f"\n📊 {len(value_picks)} picks today")
    return "\n".join(lines)
