"""Execute the explicitly authorized G3 live acceptance plan on loopback."""

from __future__ import annotations

import argparse
import http.cookiejar
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from copper_mvp.common import DEFAULT_RUNS_DIR, PROJECT_ROOT, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=PROJECT_ROOT / "runs/g3/live_validation_plan.json")
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    if plan.get("authorization") != "granted":
        parser.error("The live acceptance plan has not been explicitly authorized.")
    if len(plan["cases"]) != 4 or plan["total_cost_cap_cny"] != 2 or plan["per_task_cost_cap_cny"] != 0.5:
        parser.error("Unexpected acceptance scope.")
    output = args.plan.parent / "live_validation_results.json"
    if output.exists():
        parser.error("An acceptance record already exists; inspect it before making additional calls.")
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({}), urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar())
    )

    def call(path, data=None):
        request = urllib.request.Request(
            "http://127.0.0.1:8765" + path,
            data=json.dumps(data).encode("utf-8") if data is not None else None,
            headers={"Content-Type": "application/json"} if data is not None else {},
        )
        with opener.open(request, timeout=30) as response:
            return json.load(response)

    key = (DEFAULT_RUNS_DIR / "owner_access.key").read_text(encoding="utf-8").strip()
    call("/api/auth/session", {"access_code": key})
    info = call("/api/v2/assistant")
    if not info["live_calls_enabled"] or info["model"] != plan["provider"]:
        parser.error("The expected live provider is not enabled.")
    sessions = {}
    result = {"status": "running", "total_cost_cap_cny": 2, "tasks": []}
    write_json(output, result)
    for index, case in enumerate(plan["cases"]):
        accounted = sum(t["usage"]["spent_cny"] + t["usage"]["unsettled_cny"] for t in result["tasks"])
        if accounted + 0.5 > 2:
            raise RuntimeError("Acceptance cost cap would be exceeded.")
        if case["session"] not in sessions:
            session = call(
                "/api/v2/sessions", {"title": "G3 真实验收 · " + case["session"], "context": case["context"]}
            )
            sessions[case["session"]] = session["id"]
        task = call(
            "/api/v2/sessions/" + sessions[case["session"]] + "/messages",
            {
                "question": case["question"],
                "context": case["context"],
                "request_key": "g3-live-20260911-" + str(index),
                "max_cost_cny": 0.5,
            },
        )
        result["active_task"] = task["id"]
        write_json(output, result)
        print(json.dumps({"case": index + 1, "task_id": task["id"], "status": task["status"]}), flush=True)
        deadline = time.monotonic() + 360
        while task["status"] in ("queued", "running", "pausing", "cancelling"):
            if time.monotonic() > deadline:
                raise RuntimeError("Task still requires observation; its ID is preserved in the acceptance record.")
            time.sleep(1)
            task = call("/api/v2/tasks/" + task["id"])
        result["tasks"].append(task)
        result.pop("active_task", None)
        write_json(output, result)
        print(
            json.dumps(
                {
                    "case": index + 1,
                    "status": task["status"],
                    "usage": task["usage"],
                    "model_calls": task["model_calls"],
                    "tool_calls": task["tool_calls"],
                }
            ),
            flush=True,
        )
        if task["status"] != "completed":
            result["status"] = "needs_review"
            write_json(output, result)
            return 1
    result["status"] = "completed"
    result["spent_cny"] = sum(t["usage"]["spent_cny"] for t in result["tasks"])
    result["unsettled_cny"] = sum(t["usage"]["unsettled_cny"] for t in result["tasks"])
    write_json(output, result)
    print(
        json.dumps(
            {"status": result["status"], "spent_cny": result["spent_cny"], "unsettled_cny": result["unsettled_cny"]}
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
