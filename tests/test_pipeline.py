import numpy as np
import pandas as pd
import pytest

from predictor.data import _read_csv_text, normalise
from predictor.model import DixonColes, market_probs
from predictor.value import StrategyConfig, devig, select_bet
from tests.simulate import simulate


def test_devig_sums_to_one():
    for method in ("multiplicative", "power"):
        p = devig([2.0, 3.5, 4.0], method)
        assert p.sum() == pytest.approx(1.0)
        assert p[0] > p[1] > p[2]
    assert np.isnan(devig([2.0, np.nan, 4.0])).all()


def test_market_probs_consistent():
    m = DixonColes()
    m.teams_ = ["A", "B"]
    m.attack_, m.defence_ = np.array([0.2, -0.2]), np.array([-0.1, 0.1])
    m.mu0_, m.home_adv_, m.rho_ = 0.1, 0.25, -0.05
    p = market_probs(m.score_matrix("A", "B"))
    assert p["H"] + p["D"] + p["A"] == pytest.approx(1.0)
    assert p["O25"] + p["U25"] == pytest.approx(1.0)
    assert p["1X"] == pytest.approx(p["H"] + p["D"])
    assert p["H"] > p["A"]


def test_fit_recovers_true_parameters(tmp_path):
    truths = simulate(tmp_path, n_seasons=2, seed=3)
    truth = truths["1920"]
    df = normalise(_read_csv_text((tmp_path / "E0_1920.csv").read_text()), season="1920")
    assert len(df) == 380
    fit = DixonColes(xi=0.0, ridge=0.0).fit(df)
    assert fit.home_adv_ == pytest.approx(truth.home_adv_, abs=0.1)
    assert fit.mu0_ == pytest.approx(truth.mu0_, abs=0.1)
    order = [fit.teams_.index(t) for t in truth.teams_]
    assert np.corrcoef(fit.attack_[order], truth.attack_)[0, 1] > 0.75
    assert np.corrcoef(fit.defence_[order], truth.defence_)[0, 1] > 0.6


def test_select_bet_rules():
    row = pd.Series({"soft_H": 2.4, "soft_D": 3.3, "soft_A": 3.2, "soft_O25": 1.9, "soft_U25": 1.9,
                     "pin_H": 2.3, "pin_D": 3.4, "pin_A": 3.3, "pin_O25": 1.95, "pin_U25": 1.93,
                     "softbook_H": "B365"})
    model_p = {"H": 0.48, "D": 0.27, "A": 0.25, "O25": 0.5, "U25": 0.5}
    bet = select_bet(row, model_p, StrategyConfig(strategy="both", min_edge=0.05))
    assert bet is not None and bet["outcome"] == "H"
    assert bet["edge"] == pytest.approx(0.48 * 2.4 - 1)
    # model likes it but Pinnacle doesn't -> 'both' refuses
    model_p2 = {"H": 0.30, "D": 0.40, "A": 0.30, "O25": 0.5, "U25": 0.5}
    bet2 = select_bet(row, model_p2, StrategyConfig(strategy="both", min_edge=0.05))
    assert bet2 is None
