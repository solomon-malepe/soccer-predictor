"""Map team names from the odds API (OddsPapi) to football-data.co.uk spellings.

Matching order:
  1. ALIASES: names that can't be worked out automatically ("Inter Milano" -> "Inter")
  2. exact match after normalising (drop accents, "FC", "AFC", "Calcio", numbers...)
  3. the football-data name's words all appear in the API name ("Hull" in "Hull City"),
     if exactly one team fits
  4. close spelling (difflib), for promoted teams nobody has added yet

Anything still unmatched is reported and skipped, never guessed silently.
When a new promoted team shows up as unmatched, add it to ALIASES.
"""

from __future__ import annotations

import difflib
import re
import unicodedata

# OddsPapi name -> football-data name
ALIASES = {
    # England
    "Manchester City": "Man City",
    "Manchester United": "Man United",
    "Nottingham Forest": "Nott'm Forest",
    "Wolverhampton Wanderers": "Wolves",
    "Sheffield United": "Sheffield United",
    "West Ham United": "West Ham",
    # Spain
    "Athletic Bilbao": "Ath Bilbao",
    "Atletico Madrid": "Ath Madrid",
    "Espanyol Barcelona": "Espanol",
    "RC Deportivo de A Coruna": "La Coruna",
    "Real Valladolid": "Valladolid",
    # Italy
    "Inter Milano": "Inter",
    "Hellas Verona": "Verona",
    # Germany
    "1. FC Cologne": "FC Koln",
    "Borussia Monchengladbach": "M'gladbach",
    "Eintracht Frankfurt": "Ein Frankfurt",
    "Hamburger SV": "Hamburg",
    "FC St. Pauli": "St Pauli",
    "1. FC Heidenheim": "Heidenheim",
    "VfL Bochum": "Bochum",
    "Holstein Kiel": "Holstein Kiel",
    # France
    "Paris Saint-Germain": "Paris SG",
    "Stade Rennais FC": "Rennes",
    "AS Saint-Etienne": "St Etienne",
    "Montpellier HSC": "Montpellier",
}

_DROP = {
    "fc", "afc", "cf", "ac", "acf", "as", "ss", "ssc", "us", "bc", "cfc", "calcio", "sc", "sco",
    "ogc", "osc", "club", "de", "ud", "ca", "rc", "tsg", "vfb", "vfl", "fsv", "sv", "aj", "estac",
    "stade", "alsace", "hsc", "the",
}


def normalise(name: str) -> str:
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-z0-9 ]", " ", s.lower().replace("&", " "))
    words = [w for w in s.split() if w not in _DROP and not w.isdigit()]
    return " ".join(words)


def match_team(api_name: str, fd_names) -> str | None:
    """Best football-data name for an API team name, or None if no safe match."""
    fd_names = list(fd_names)
    if api_name in ALIASES and ALIASES[api_name] in fd_names:
        return ALIASES[api_name]
    if api_name in fd_names:
        return api_name
    target = normalise(api_name)
    norm = {fd: normalise(fd) for fd in fd_names}
    exact = [fd for fd, n in norm.items() if n == target]
    if len(exact) == 1:
        return exact[0]
    words = set(target.split())
    subset = [fd for fd, n in norm.items() if n and set(n.split()) <= words]
    if len(subset) == 1:
        return subset[0]
    close = difflib.get_close_matches(target, list(norm.values()), n=2, cutoff=0.85)
    if len(close) == 1:
        return next(fd for fd, n in norm.items() if n == close[0])
    return None


def map_names(api_names, fd_names) -> tuple[dict[str, str], list[str]]:
    """Map many API names at once. Returns (mapping, unmatched). Two API teams
    mapping to the same football-data team counts as unmatched for both."""
    mapping = {a: match_team(a, fd_names) for a in set(api_names)}
    taken: dict[str, list[str]] = {}
    for a, fd in mapping.items():
        if fd:
            taken.setdefault(fd, []).append(a)
    clashes = {a for names in taken.values() if len(names) > 1 for a in names}
    unmatched = sorted(a for a, fd in mapping.items() if fd is None or a in clashes)
    return {a: fd for a, fd in mapping.items() if fd and a not in clashes}, unmatched
