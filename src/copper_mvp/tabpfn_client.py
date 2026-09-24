"""One local TabPFN worker per host process; only its current context is cached."""

import atexit
import json
import os
import queue
import subprocess
import sys
import threading
import uuid
from contextlib import contextmanager

from copper_mvp.common import PROJECT_ROOT, WorkbenchError

_LOCK = threading.RLock()
_CLIENT = None


def read_responses(stream, responses):
    try:
        for line in iter(stream.readline, b""):
            text = line.decode("utf-8", errors="replace")
            if text.startswith("TABPFN_JSON "):
                responses.put(json.loads(text[len("TABPFN_JSON ") :]))
    finally:
        responses.put(None)


class LocalWorker:
    def __init__(self):
        root = PROJECT_ROOT / "runs/mvp/tabpfn_worker_logs"
        root.mkdir(parents=True, exist_ok=True)
        self.stderr = (root / (uuid.uuid4().hex + ".log")).open("wb")
        flags = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
        self.process = subprocess.Popen(
            [sys.executable, "-X", "utf8", "-B", str(PROJECT_ROOT / "scripts/run_tabpfn_worker.py")],
            cwd=PROJECT_ROOT,
            env=os.environ.copy(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.stderr,
            **flags,
        )
        self.responses = queue.Queue()
        threading.Thread(target=read_responses, args=(self.process.stdout, self.responses), daemon=True).start()

    def call(self, request, timeout):
        identifier = uuid.uuid4().hex
        self.process.stdin.write((json.dumps({**request, "request_id": identifier}) + "\n").encode())
        self.process.stdin.flush()
        try:
            response = self.responses.get(timeout=timeout)
        except queue.Empty:
            self.close()
            raise WorkbenchError("TabPFN 达到计算时间预算", "TABPFN_TIME_BUDGET") from None
        if response is None:
            raise WorkbenchError("TabPFN 独立进程已退出", "TABPFN_WORKER_EXITED")
        if response.get("request_id") != identifier:
            raise WorkbenchError("TabPFN 响应身份不匹配", "TABPFN_RESPONSE")
        if response.get("status") != "ok":
            raise WorkbenchError("TabPFN 计算失败: " + response.get("error", "unknown"), "TABPFN_WORKER_ERROR")
        return response["result"]

    def close(self):
        if self.process.poll() is None:
            try:
                self.process.stdin.close()
                self.process.wait(timeout=5)
            except (OSError, subprocess.TimeoutExpired):
                self.process.kill()
                self.process.wait()
        self.stderr.close()


def request_worker(request, timeout=1800):
    global _CLIENT
    with _LOCK:
        if _CLIENT is None or _CLIENT.process.poll() is not None:
            _CLIENT = LocalWorker()
        return _CLIENT.call(request, timeout)


def close_worker():
    global _CLIENT
    with _LOCK:
        if _CLIENT is not None:
            _CLIENT.close()
            _CLIENT = None


atexit.register(close_worker)


@contextmanager
def worker_transaction():
    with _LOCK:
        yield
