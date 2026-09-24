"""Only the independent loopback Mock protocol is reachable through this client."""

from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import urlparse

import httpx

from copper_mvp.common import WorkbenchError
from copper_mvp.control.contracts import DEVICE, POLICY


class MockClient:
    def __init__(self, url, driver_key, admin_key=None, *, client=None):
        parsed = urlparse(url)
        if (
            parsed.scheme != "http"
            or parsed.hostname not in ("127.0.0.1", "localhost", "::1")
            or parsed.username
            or parsed.password
            or parsed.path not in ("", "/")
            or parsed.query
        ):
            raise WorkbenchError("Mock 端点必须是本机独立 HTTP 服务", "MOCK_ENDPOINT")
        self.url = url.rstrip("/")
        self.driver_key, self.admin_key = driver_key, admin_key
        self.client = client or httpx.Client(timeout=3, trust_env=False, follow_redirects=False)

    @classmethod
    def from_env(cls):
        url, path = os.environ.get("COPPER_MOCK_URL"), os.environ.get("COPPER_MOCK_KEY_FILE")
        if not url or not path:
            return None
        admin = os.environ.get("COPPER_MOCK_ADMIN_KEY_FILE")
        return cls(
            url,
            Path(path).read_text(encoding="utf-8").strip(),
            Path(admin).read_text(encoding="utf-8").strip() if admin else None,
        )

    def close(self):
        self.client.close()

    def request(self, method, path, body=None, admin=False):
        key = self.admin_key if admin else self.driver_key
        if not key:
            raise WorkbenchError("未配置 Mock 测试管理员凭据", "FORBIDDEN")
        try:
            response = self.client.request(
                method, self.url + path, json=body, headers={"Authorization": "Bearer " + key}
            )
        except httpx.TransportError as exc:
            raise WorkbenchError("Mock 连接中断，已发送命令需查询账本协调", "MOCK_TRANSPORT") from exc
        if response.status_code >= 500:
            raise WorkbenchError("Mock 未返回确定结果，需查询账本协调", "MOCK_TRANSPORT")
        if response.status_code >= 400:
            error = response.json().get("error", {})
            raise WorkbenchError(error.get("message", "Mock 拒绝请求"), error.get("code", "MOCK_REJECTED"))
        return response.json()

    def read_state(self):
        value = self.request("GET", "/v1/state")
        if value.get("environment") != "MOCK" or value.get("policy_ref") != POLICY or value.get("device_id") != DEVICE:
            raise WorkbenchError("端点未提供约定的合成设备协议", "MOCK_POLICY")
        return value

    def read_points(self):
        return self.request("GET", "/v1/points")

    def validate_command(self, command):
        return self.request("POST", "/v1/validate", command)

    def submit_command(self, command, payload_hash, fencing_token, lease_until):
        return self.request(
            "POST",
            "/v1/commands",
            {
                "command": command,
                "payload_hash": payload_hash,
                "fencing_token": fencing_token,
                "lease_until": lease_until,
            },
        )

    def get_command_status(self, command_id):
        return self.request("GET", "/v1/commands/" + command_id)["record"]

    def tick(self, seconds):
        return self.request("POST", "/v1/test-admin/tick", {"seconds": seconds}, admin=True)

    def inject_fault(self, fault):
        return self.request("POST", "/v1/test-admin/faults", fault, admin=True)
