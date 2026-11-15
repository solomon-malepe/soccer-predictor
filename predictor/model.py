"""Dixon-Coles (1997) model with exponential time decay.

    log(lambda_home) = mu0 + home_adv + attack[home] + defence[away]
    log(lambda_away) = mu0 +            attack[away] + defence[home]

attack and defence are centred to sum to zero. A positive defence value means
a team concedes MORE than average. rho corrects the low-score cells (0-0, 1-0,
0-1, 1-1) that a plain Poisson model gets wrong.

Matches are weighted by exp(-xi * days_ago), so recent form counts more.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

DEFAULT_XI = 0.0019  # half-life of roughly one year


@dataclass
class DixonColes:
    xi: float = DEFAULT_XI
    ridge: float = 3.0        # pulls ratings toward average; stops promoted teams with few games looking extreme
    max_goals: int = 10

    teams_: list[str] = field(default_factory=list, init=False)
    attack_: np.ndarray | None = field(default=None, init=False)
    defence_: np.ndarray | None = field(default=None, init=False)
    mu0_: float = field(default=0.0, init=False)
    home_adv_: float = field(default=0.0, init=False)
    rho_: float = field(default=0.0, init=False)
    match_counts_: dict[str, float] = field(default_factory=dict, init=False)
    fitted_at_: pd.Timestamp | None = field(default=None, init=False)

    # ------------------------------------------------------------------ fit
    def fit(self, matches: pd.DataFrame, as_of: pd.Timestamp | None = None,
            warm_start: "DixonColes | None" = None) -> "DixonColes":
        """Fit on matches with columns date, home, away, hg, ag (played games only)."""
        m = matches.dropna(subset=["hg", "ag"])
        if as_of is not None:
            m = m[m["date"] < as_of]
        if len(m) < 30:
            raise ValueError(f"Need at least 30 played matches to fit, got {len(m)}")
        as_of = as_of if as_of is not None else m["date"].max() + pd.Timedelta(days=1)

        teams = sorted(set(m["home"]) | set(m["away"]))
        idx = {t: i for i, t in enumerate(teams)}
        n = len(teams)
        h = m["home"].map(idx).to_numpy()
        a = m["away"].map(idx).to_numpy()
        x = m["hg"].to_numpy(dtype=float)
        y = m["ag"].to_numpy(dtype=float)
        days = (as_of - m["date"]).dt.days.to_numpy(dtype=float)
        w = np.exp(-self.xi * days)

        # masks for the Dixon-Coles low-score cells
        m00 = (x == 0) & (y == 0)
        m01 = (x == 0) & (y == 1)
        m10 = (x == 1) & (y == 0)
        m11 = (x == 1) & (y == 1)

        def unpack(p):
            att_raw, def_raw = p[:n], p[n:2 * n]
            mu0, home, rho = p[2 * n], p[2 * n + 1], p[2 * n + 2]
            return att_raw - att_raw.mean(), def_raw - def_raw.mean(), mu0, home, rho, att_raw, def_raw

        def nll_and_grad(p):
            att, dfn, mu0, home, rho, att_raw, def_raw = unpack(p)
            log_lam = mu0 + home + att[h] + dfn[a]
            log_mu = mu0 + att[a] + dfn[h]
            lam, mu = np.exp(log_lam), np.exp(log_mu)

            tau = np.ones_like(lam)
            tau[m00] = 1 - lam[m00] * mu[m00] * rho
            tau[m01] = 1 + lam[m01] * rho
            tau[m10] = 1 + mu[m10] * rho
            tau[m11] = 1 - rho
            tau = np.maximum(tau, 1e-10)

            ll = w * (np.log(tau) + x * log_lam - lam + y * log_mu - mu)
            penalty = self.ridge * (np.sum(att_raw ** 2) + np.sum(def_raw ** 2))
            nll = -ll.sum() + penalty

            # gradient w.r.t. log_lam, log_mu and rho
            dtau_dloglam = np.zeros_like(lam)
            dtau_dlogmu = np.zeros_like(lam)
            dtau_drho = np.zeros_like(lam)
            dtau_dloglam[m00] = -lam[m00] * mu[m00] * rho
            dtau_dlogmu[m00] = -lam[m00] * mu[m00] * rho
            dtau_drho[m00] = -lam[m00] * mu[m00]
            dtau_dloglam[m01] = lam[m01] * rho
            dtau_drho[m01] = lam[m01]
            dtau_dlogmu[m10] = mu[m10] * rho
            dtau_drho[m10] = mu[m10]
            dtau_drho[m11] = -1.0

            g_lam = w * (x - lam + dtau_dloglam / tau)
            g_mu = w * (y - mu + dtau_dlogmu / tau)
            g_rho = np.sum(w * dtau_drho / tau)

            g_att = np.bincount(h, g_lam, n) + np.bincount(a, g_mu, n)
            g_def = np.bincount(a, g_lam, n) + np.bincount(h, g_mu, n)
            # chain rule through the centring step, plus ridge
            g_att_raw = -(g_att - g_att.mean()) + 2 * self.ridge * att_raw
            g_def_raw = -(g_def - g_def.mean()) + 2 * self.ridge * def_raw
            grad = np.concatenate([
                g_att_raw, g_def_raw,
                [-(g_lam.sum() + g_mu.sum()), -g_lam.sum(), -g_rho],
            ])
            return nll, grad

        p0 = np.zeros(2 * n + 3)
        p0[2 * n] = np.log(max(np.average(np.r_[x, y], weights=np.r_[w, w]), 0.1))
        p0[2 * n + 1] = 0.25
        if warm_start is not None and warm_start.attack_ is not None:
            old = {t: i for i, t in enumerate(warm_start.teams_)}
            for t, i in idx.items():
                if t in old:
                    p0[i] = warm_start.attack_[old[t]]
                    p0[n + i] = warm_start.defence_[old[t]]
            p0[2 * n:] = [warm_start.mu0_, warm_start.home_adv_, warm_start.rho_]

        bounds = [(None, None)] * (2 * n + 2) + [(-0.25, 0.25)]
        res = minimize(nll_and_grad, p0, jac=True, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": 500})
        att, dfn, mu0, home, rho, _, _ = unpack(res.x)

        self.teams_ = teams
        self.attack_, self.defence_ = att, dfn
        self.mu0_, self.home_adv_, self.rho_ = float(mu0), float(home), float(rho)
        counts = np.bincount(h, w, n) + np.bincount(a, w, n)
        raw_counts = np.bincount(h, minlength=n) + np.bincount(a, minlength=n)
        self.match_counts_ = {t: int(raw_counts[i]) for t, i in idx.items()}
        self._weighted_counts = {t: float(counts[i]) for t, i in idx.items()}
        self.fitted_at_ = as_of
        self.converged_ = bool(res.success)
        return self

    # ------------------------------------------------------------- predict
    def knows(self, team: str, min_matches: int = 1) -> bool:
        return self.match_counts_.get(team, 0) >= min_matches

    def expected_goals(self, home: str, away: str) -> tuple[float, float]:
        i, j = self.teams_.index(home), self.teams_.index(away)
        lam = np.exp(self.mu0_ + self.home_adv_ + self.attack_[i] + self.defence_[j])
        mu = np.exp(self.mu0_ + self.attack_[j] + self.defence_[i])
        return float(lam), float(mu)

    def score_matrix(self, home: str, away: str) -> np.ndarray:
        """P(home scores i, away scores j) for i, j in 0..max_goals."""
        lam, mu = self.expected_goals(home, away)
        g = np.arange(self.max_goals + 1)
        mat = np.outer(poisson.pmf(g, lam), poisson.pmf(g, mu))
        rho = self.rho_
        mat[0, 0] *= 1 - lam * mu * rho
        mat[0, 1] *= 1 + lam * rho
        mat[1, 0] *= 1 + mu * rho
        mat[1, 1] *= 1 - rho
        return mat / mat.sum()

    def ratings(self) -> pd.DataFrame:
        """Team ratings table, strongest first."""
        df = pd.DataFrame({
            "team": self.teams_,
            "attack": self.attack_,
            "defence": self.defence_,
            "matches": [self.match_counts_[t] for t in self.teams_],
        })
        df["overall"] = df["attack"] - df["defence"]
        return df.sort_values("overall", ascending=False).reset_index(drop=True)


def market_probs(mat: np.ndarray) -> dict[str, float]:
    """Every market we care about, priced from one score matrix."""
    g = np.arange(mat.shape[0])
    total = g[:, None] + g[None, :]
    home_win = float(np.tril(mat, -1).sum())
    draw = float(np.trace(mat))
    away_win = float(np.triu(mat, 1).sum())
    p = {
        "H": home_win, "D": draw, "A": away_win,
        "1X": home_win + draw, "X2": draw + away_win, "12": home_win + away_win,
        "BTTS_Y": float(mat[1:, 1:].sum()),
    }
    p["BTTS_N"] = 1 - p["BTTS_Y"]
    for line in (1.5, 2.5, 3.5):
        key = str(line).replace(".", "")
        over = float(mat[total > line].sum())
        p[f"O{key}"], p[f"U{key}"] = over, 1 - over
    return p
