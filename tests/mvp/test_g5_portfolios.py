"""Budget, state, archive and independent objective checks for optimizer portfolios."""

import json

import numpy as np
import pandas as pd
import pytest

from copper_mvp.common import WorkbenchError
from copper_mvp.contracts import RunRequest
from copper_mvp.optimization_problem import build_problem
from copper_mvp.optimizer_comparison import run_optimizer
from copper_mvp.optimizer_portfolio import (
    STRATEGIES,
    PortfolioEvaluator,
    import_archive,
    merge_archives,
    run_portfolio,
    signed_archive,
)


def problem():
    return build_problem(None, None, RunRequest(task_type="optimize", mode="benchmark", request_key="portfolio-test"))


def independent_hv(front, reference):
    total = 0.0
    previous = reference[1]
    for x, y in front[np.argsort(front[:, 0])]:
        if x <= reference[0] and y < previous:
            total += (reference[0] - x) * (previous - y)
            previous = y
    return total


def test_shared_budget_pilots_and_independent_front_verification(tmp_path):
    results = []
    for strategy in STRATEGIES:
        root = tmp_path / strategy
        result = run_portfolio(problem(), strategy, 31, 512, 15, root)
        results.append(result)
        frame = pd.read_csv(root / "evaluations.csv")
        assert len(frame) == result["total_evaluations"] == sum(result["evaluations"].values()) <= 512
        X = frame[["x0", "x1"]].to_numpy()
        assert np.allclose(frame[["f0", "f1"]], np.column_stack(((X * X).sum(1), ((X - 2) ** 2).sum(1))))
        assert np.allclose(frame.g0, X.sum(1) - 3)
        front = np.array([[c["f1"], c["f2"]] for c in result["candidates"]]) / np.array(result["metric"]["scale"])
        assert result["hv"] == pytest.approx(independent_hv(front, np.array([1.1, 1.1])))
        assert all(max(c["constraints"]) <= 1e-8 for c in result["candidates"])
        assert result["evaluations"]["verification"] == len(result["candidates"])
        if strategy in ("adaptive", "equal_share"):
            trace = json.loads((root / "allocation_trace.json").read_text(encoding="utf-8"))
            pilots = [r for r in trace if r["stage"] == "arm_pilot"]
            assert {r["arm"] for r in pilots} == {"NSGA-II", "SPEA2"}
            assert (
                len({sum(r["charged_evaluations"] for r in pilots if r["arm"] == arm) for arm in ("NSGA-II", "SPEA2")})
                == 1
            )
            assert all(r["archive_points_received"] == 0 and r["credit_basis"] == "own_archive" for r in pilots)
    assert len({r["problem_signature"] for r in results}) == 1
    assert len({r["initial_population_hash"] for r in results}) == 1


def test_fixed_strategies_reproduce_existing_optimizer_search(tmp_path):
    for strategy, method in (("fixed_nsga2", "NSGA-II"), ("fixed_spea2", "SPEA2")):
        old = run_optimizer(problem(), method, 17, 256, 15, tmp_path / (strategy + "-old"))
        new = run_portfolio(problem(), strategy, 17, 256, 15, tmp_path / (strategy + "-new"))
        assert old["candidates"] == new["candidates"]
        assert old["hv"] == new["hv"]
        assert old["total_evaluations"] == new["total_evaluations"]


def test_archive_signature_change_recomputes_and_tamper_is_rejected():
    prepared = problem()
    meter = PortfolioEvaluator(prepared, 64)
    X = np.array([[0.5, 0.5], [1.0, 1.0]])
    F, G, _ = prepared.evaluate(X)
    archive = signed_archive("old", X, F, G)
    same, proof = import_archive(archive, "old", meter)
    assert proof == {"re_evaluated": 0, "reused": 2} and meter.used == 1
    new, proof = import_archive(archive, "new", meter)
    assert proof == {"re_evaluated": 2, "reused": 0} and meter.used == 3
    assert np.allclose(new.get("F"), F)
    with pytest.raises(WorkbenchError, match="签名"):
        merge_archives([archive], "new")
    altered = {**archive, "F": [[999.0, 999.0], [999.0, 999.0]]}
    with pytest.raises(WorkbenchError, match="哈希"):
        import_archive(altered, "old", meter)


def test_infeasible_plateau_and_new_problem_do_not_share_algorithm_state(tmp_path):
    original = run_portfolio(problem(), "adaptive", 17, 256, 15, tmp_path / "original")
    for name, violation in (("infeasible", 1.0), ("plateau", -1.0)):
        prepared = problem()
        prepared.evaluate = lambda X, g=violation: (np.ones((len(X), 2)), np.full((len(X), 1), g), {})
        prepared.ref_F = np.ones((1, 2))
        prepared.ref_G = np.full((1, 1), violation)
        prepared.reference_front = None
        spec = prepared.specification()
        prepared.specification = lambda s=spec, n=name: {**s, "fixture": n}
        result = run_portfolio(prepared, "adaptive", 91, 256, 15, tmp_path / name)
        assert result["total_evaluations"] <= 256
        if name == "infeasible":
            assert result["status"] == "infeasible" and result["hv"] == 0 and not result["candidates"]
        else:
            assert result["objective_front_points"] == 1
            assert result["decision_front_points"] > 1
    repeated = run_portfolio(problem(), "adaptive", 17, 256, 15, tmp_path / "repeat")
    assert original["candidates"] == repeated["candidates"] and original["arms"] == repeated["arms"]


def test_small_budget_falls_back_without_unfunded_arm_pilots(tmp_path):
    result = run_portfolio(problem(), "adaptive", 17, 128, 15, tmp_path)
    assert set(result["arms"]) == {"NSGA-II"}
    assert result["total_evaluations"] <= 128


def test_comparison_reuses_verified_runs_and_rejects_changed_protocol(tmp_path):
    from copper_mvp.portfolio_comparison import PortfolioRequest, compare_portfolios

    request = PortfolioRequest(request_key="comparison", mode="benchmark", seeds=(17, 29), total_budget=256)
    first = compare_portfolios(None, None, request, tmp_path)
    assert first["runs"] == 8 and first["failed_runs"] == 0
    assert first["paired"]["fixed_nsga2"]["ties"] == 2
    hashes = first["run_hashes"]
    second = compare_portfolios(None, None, request, tmp_path)
    assert second["run_hashes"] == hashes
    changed = request.model_copy(update={"total_budget": 512})
    with pytest.raises(WorkbenchError, match="已变化"):
        compare_portfolios(None, None, changed, tmp_path)


def test_portfolio_api_identity_and_saved_run_evidence(tmp_path, monkeypatch):
    import time

    from fastapi.testclient import TestClient

    from copper_mvp.api import create_app

    monkeypatch.delenv("COPPER_MOCK_URL", raising=False)
    monkeypatch.delenv("COPPER_MOCK_KEY_FILE", raising=False)
    monkeypatch.setenv("COPPER_ASSISTANT_LIVE_CALLS", "0")
    with TestClient(create_app(run_dir=tmp_path / "api"), base_url="http://127.0.0.1") as client:
        wb = client.app.state.workbench
        assert client.get("/api/v2/optimizer-portfolios").status_code == 401
        key = wb.access.owner_key_path.read_text(encoding="utf-8").strip()
        client.post("/api/auth/session", json={"access_code": key})
        response = client.post(
            "/api/v2/optimizer-portfolios",
            json={"request_key": "api-math", "mode": "benchmark", "seeds": [17], "total_budget": 128},
        )
        assert response.status_code == 202
        identifier = response.json()["id"]
        for _ in range(100):
            state = client.get("/api/v2/optimizer-portfolios/" + identifier).json()
            if state["status"] not in ("queued", "running"):
                break
            time.sleep(0.05)
        assert state["status"] == "completed" and state["result"]["runs"] == 4
        run = client.get(
            "/api/v2/optimizer-portfolios/" + identifier + "/runs",
            params={"case": 0, "seed": 17, "strategy": "adaptive"},
        )
        assert run.status_code == 200 and run.json()["total_evaluations"] <= 128
        assert client.get("/api/v2/optimizer-portfolios/" + identifier + "/export").status_code == 200
        assert client.post("/api/v2/optimizer-portfolios/" + identifier + "/resume", json={}).status_code == 409
        actor = wb.access.authenticate(key=key)
        viewer = wb.access.issue(actor, "viewer", "viewer")
        headers = {"Authorization": "Bearer " + viewer}
        assert client.get("/api/v2/optimizer-portfolios", headers=headers).json()["items"] == []
        assert client.get("/api/v2/optimizer-portfolios/" + identifier, headers=headers).status_code == 403
        assert (
            client.post(
                "/api/v2/optimizer-portfolios",
                headers=headers,
                json={"request_key": "denied", "mode": "benchmark", "seeds": [17]},
            ).status_code
            == 403
        )


def test_failed_attempt_is_preserved_after_recovery_and_not_counted_as_fair_pair(tmp_path, monkeypatch):
    from copper_mvp.common import write_json
    from copper_mvp.optimizer_portfolio import run_portfolio as actual
    from copper_mvp.portfolio_comparison import PortfolioRequest, compare_portfolios

    state = {"failed": False}

    def once(prepared, strategy, seed, budget, seconds, output, **kwargs):
        if strategy == "adaptive" and not state["failed"]:
            state["failed"] = True
            output.mkdir(parents=True, exist_ok=True)
            write_json(
                output / "failure.json", {"total_evaluations": 7, "charged_evaluations": {"reference": 1, "pilot": 6}}
            )
            raise WorkbenchError("synthetic transient failure", "SYNTHETIC_FAILURE")
        return actual(prepared, strategy, seed, budget, seconds, output, **kwargs)

    monkeypatch.setattr("copper_mvp.portfolio_comparison.run_portfolio", once)
    request = PortfolioRequest(request_key="failure", mode="benchmark", seeds=(17,), total_budget=128)
    first = compare_portfolios(None, None, request, tmp_path)
    assert first["failed_runs"] == 1
    second = compare_portfolios(None, None, request, tmp_path)
    recovered = next(r for r in second["results"] if r["strategy"] == "adaptive")
    assert recovered["status"] == "feasible" and len(recovered["attempts"]) == 2
    assert recovered["attempts"][0]["total_evaluations"] == 7
    assert second["paired"]["adaptive"]["failed_or_unpaired"] == 1
    assert second["total_recorded_evaluations"] == sum(
        a["total_evaluations"] or 0 for r in second["results"] for a in r["attempts"]
    )
