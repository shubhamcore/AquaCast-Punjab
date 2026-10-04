"""
AquaCast-Punjab :: Bilingual (English / Punjabi-Gurmukhi) alert templates
=========================================================================

Every farmer-facing string lives here so the advisory engine stays clean and the
language layer can be swapped or extended (Hindi, Saraiki) without touching logic.
Punjabi is rendered in Gurmukhi — the script actually used on Punjab handsets.
"""
from __future__ import annotations

ZONES = {
    "safe": {
        "en": "SAFE", "pa": "ਸੁਰੱਖਿਅਤ",
        "colour": "#22C55E", "emoji": "🟢",
    },
    "critical": {
        "en": "CRITICAL", "pa": "ਨਾਜ਼ੁਕ",
        "colour": "#F59E0B", "emoji": "🟠",
    },
    "over_exploited": {
        "en": "OVER-EXPLOITED", "pa": "ਵੱਧ-ਖਿੱਚ",
        "colour": "#EF4444", "emoji": "🔴",
    },
}

BAND = {
    "low":      {"en": "Low", "pa": "ਘੱਟ", "colour": "#22C55E"},
    "moderate": {"en": "Moderate", "pa": "ਦਰਮਿਆਨਾ", "colour": "#84CC16"},
    "elevated": {"en": "Elevated", "pa": "ਵਧਿਆ", "colour": "#F59E0B"},
    "high":     {"en": "High", "pa": "ਉੱਚਾ", "colour": "#F97316"},
    "severe":   {"en": "Severe", "pa": "ਗੰਭੀਰ", "colour": "#EF4444"},
}

LABELS = {
    "water_level":      {"en": "Water table", "pa": "ਪਾਣੀ ਦਾ ਪੱਧਰ"},
    "metres_below":     {"en": "m below ground", "pa": "ਮੀਟਰ ਹੇਠਾਂ"},
    "at_harvest":       {"en": "Predicted at harvest", "pa": "ਵਾਢੀ ਸਮੇਂ ਅਨੁਮਾਨ"},
    "pump_hours":       {"en": "Pump quota", "pa": "ਟਿਊਬਵੈੱਲ ਸਮਾਂ"},
    "hours_per_week":   {"en": "hours / week / acre", "pa": "ਘੰਟੇ / ਹਫ਼ਤਾ / ਏਕੜ"},
    "pump_risk":        {"en": "Pump failure risk", "pa": "ਪੰਪ ਖ਼ਰਾਬ ਹੋਣ ਦਾ ਖ਼ਤਰਾ"},
    "zone":             {"en": "Aquifer zone", "pa": "ਧਰਤੀ ਹੇਠਲੇ ਪਾਣੀ ਦੀ ਹਾਲਤ"},
    "irrigate_now":     {"en": "Irrigate now", "pa": "ਹੁਣੇ ਸਿੰਚਾਈ ਕਰੋ"},
    "delay":            {"en": "Delay irrigation", "pa": "ਸਿੰਚਾਈ ਟਾਲੋ"},
    "advisory":         {"en": "Advisory", "pa": "ਸਲਾਹ"},
    "wheat":            {"en": "Wheat", "pa": "ਕਣਕ"},
    "rain_expected":    {"en": "Rain expected", "pa": "ਮੀਂਹ ਦੀ ਸੰਭਾਵਨਾ"},
    "save_water":       {"en": "Save water", "pa": "ਪਾਣੀ ਬਚਾਓ"},
    "credit":           {"en": "Loan risk", "pa": "ਕਰਜ਼ੇ ਦਾ ਖ਼ਤਰਾ"},
    "days_to_dryout":   {"en": "Days to pump dry-out", "pa": "ਪੰਪ ਸੁੱਕਣ ਤੱਕ ਦਿਨ"},
    "hello_farmer":     {"en": "Dear farmer", "pa": "ਪਿਆਰੇ ਕਿਸਾਨ ਵੀਰ"},
    "aqua_alert":       {"en": "AquaCast Alert", "pa": "ਐਕੁਆਕਾਸਟ ਚੇਤਾਵਨੀ"},
    "do_not_overpump":  {"en": "Do not over-pump", "pa": "ਵੱਧ ਪੰਪਿੰਗ ਨਾ ਕਰੋ"},
    "shift_drip":       {"en": "Shift to drip / sprinkler", "pa": "ਤੁਪਕਾ ਸਿੰਚਾਈ ਅਪਣਾਓ"},
    "laser_levelling":  {"en": "Use laser land levelling", "pa": "ਲੇਜ਼ਰ ਲੈਵਲਿੰਗ ਕਰੋ"},
    "contact_kisan":    {"en": "Contact your Kisan Mitra", "pa": "ਆਪਣੇ ਕਿਸਾਨ ਮਿੱਤਰ ਨਾਲ ਸੰਪਰਕ ਕਰੋ"},
    "helpline":         {"en": "Helpline", "pa": "ਹੈਲਪਲਾਈਨ"},
    "stop":             {"en": "Reply STOP to opt out", "pa": "ਬੰਦ ਕਰਨ ਲਈ STOP ਭੇਜੋ"},
}

ACTION_COPY = {
    "irrigate": {
        "en": "Irrigate now — crop is at a water-sensitive stage.",
        "pa": "ਹੁਣੇ ਸਿੰਚਾਈ ਕਰੋ — ਫ਼ਸਲ ਪਾਣੀ-ਸੰਵੇਦਨਸ਼ੀਲ ਪੜਾਅ 'ਤੇ ਹੈ।",
    },
    "hold": {
        "en": "Hold irrigation — rainfall is likely in the next 72 hours.",
        "pa": "ਸਿੰਚਾਈ ਰੋਕੋ — ਅਗਲੇ 72 ਘੰਟਿਆਂ ਵਿੱਚ ਮੀਂਹ ਦੀ ਸੰਭਾਵਨਾ ਹੈ।",
    },
    "reduce": {
        "en": "Cut pumping — the aquifer is below its safe threshold.",
        "pa": "ਪੰਪਿੰਗ ਘਟਾਓ — ਧਰਤੀ ਹੇਠਲਾ ਪਾਣੀ ਸੁਰੱਖਿਅਤ ਪੱਧਰ ਤੋਂ ਹੇਠਾਂ ਹੈ।",
    },
    "drip": {
        "en": "Shift to drip / sprinkler to cut draft by up to 40%.",
        "pa": "ਤੁਪਕਾ / ਛਿੜਕਾਅ ਸਿੰਚਾਈ ਅਪਣਾਓ — ਖਿੱਚ 40% ਤੱਕ ਘਟੇਗੀ।",
    },
    "deepen": {
        "en": "Pump dry-out likely before harvest — plan an alternate source.",
        "pa": "ਵਾਢੀ ਤੋਂ ਪਹਿਲਾਂ ਪੰਪ ਸੁੱਕਣ ਦਾ ਖ਼ਤਰਾ — ਬਦਲਵੇਂ ਸਰੋਤ ਦੀ ਯੋਜਨਾ ਬਣਾਓ।",
    },
    "credit_watch": {
        "en": "Your loan account is flagged for water-risk monitoring.",
        "pa": "ਤੁਹਾਡਾ ਕਰਜ਼ਾ ਖਾਤਾ ਪਾਣੀ-ਜੋਖਮ ਨਿਗਰਾਨੀ ਹੇਠ ਹੈ।",
    },
    "ok": {
        "en": "Aquifer is within safe limits — maintain the schedule.",
        "pa": "ਧਰਤੀ ਹੇਠਲਾ ਪਾਣੀ ਸੁਰੱਖਿਅਤ ਹੈ — ਸਮਾਂ-ਸਾਰਣੀ ਜਾਰੀ ਰੱਖੋ।",
    },
}


def pick(key: str, lang: str = "en") -> str:
    """Look up a label in the requested language (falls back to English)."""
    d = LABELS.get(key)
    if not d:
        return key
    return d.get(lang, d["en"])


def zone_key(level_mbgl: float) -> str:
    from .config import ZONE_CRITICAL_MAX, ZONE_SAFE_MAX
    if level_mbgl < ZONE_SAFE_MAX:
        return "safe"
    if level_mbgl <= ZONE_CRITICAL_MAX:
        return "critical"
    return "over_exploited"


def band_key(score: float) -> str:
    if score < 25:
        return "low"
    if score < 45:
        return "moderate"
    if score < 62:
        return "elevated"
    if score < 78:
        return "high"
    return "severe"
