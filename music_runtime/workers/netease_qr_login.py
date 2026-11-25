"""Fixed child for Netease QR login; never writes secrets to stdout/stderr."""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener


MAX_BODY = 512 * 1024
PROFILE_RE = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
UA = "NachoBot-Music-Library/1.0"
_COOKIE_NAME_RE = re.compile(r"^[!#$%&'*+\-.^_`|~0-9A-Za-z]+$")
_COOKIE_ATTRIBUTES = {
    "domain", "expires", "httponly", "max-age", "partitioned", "path", "priority",
    "sameparty", "samesite", "secure", "sourceport", "sourcescheme",
}
_ERROR_MESSAGES = {
    "qr_request_failed": "网易云音乐二维码申请失败，请检查登录服务后重试。",
    "qr_check_unavailable": "网易云音乐登录状态服务暂时不可用，请稍后重试。",
    "qr_check_failed": "二维码登录状态查询失败，请重新申请二维码。",
    "account_validation_unavailable": "登录已确认，但账号验证服务暂时不可用，请稍后重试。",
    "account_validation_failed": "账号验证失败，请重新进行网易云音乐登录。",
    "internal_error": "网易云音乐登录失败，请检查服务后重试。",
}


class LoginFailure(RuntimeError):
    """A failure with a safe, fixed status code for the UI."""

    def __init__(self, code: str):
        self.code = code if code in _ERROR_MESSAGES else "internal_error"
        super().__init__(self.code)


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Login cookies must only reach the configured loopback endpoint.
        return None


def normalize_cookie(cookie: str) -> str:
    """Keep cookie pairs and discard Set-Cookie attributes from the pinned API."""
    if not isinstance(cookie, str) or len(cookie) > 16 * 1024 or any(char in cookie for char in "\r\n\x00"):
        return ""
    pairs: dict[str, str] = {}
    for item in cookie.split(";"):
        name, separator, value = item.strip().partition("=")
        name = name.strip()
        value = value.strip()
        if not separator or not value or name.casefold() in _COOKIE_ATTRIBUTES:
            continue
        if not _COOKIE_NAME_RE.fullmatch(name) or any(ord(char) < 0x20 or ord(char) == 0x7F for char in value):
            return ""
        pairs[name] = value
    return "; ".join(f"{name}={value}" for name, value in pairs.items())


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile("wb", prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False) as output:
            temp_path = Path(output.name)
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temp_path, path)
        temp_path = None
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def _publish(status_path: Path, state: str, message: str, *, error_code: str | None = None) -> None:
    allowed = {"starting", "waiting", "scanned", "success", "expired", "error"}
    if state not in allowed:
        state, message = "error", "Netease login failed."
    payload_data = {"status": state, "message": message[:160], "updated_at": time.time()}
    if state == "error":
        error_code = error_code if isinstance(error_code, str) and error_code in _ERROR_MESSAGES else "internal_error"
        payload_data["error_code"] = error_code
        payload_data["message"] = _ERROR_MESSAGES[error_code]
    payload = json.dumps(payload_data, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    _atomic_write(status_path, payload)


def _request(base_url: str, endpoint: str, params: dict | None = None, *, cookie: str = "") -> dict:
    query = f"?{urlencode(params or {})}" if params else ""
    headers = {"Accept": "application/json", "User-Agent": UA}
    if cookie:
        headers["Cookie"] = cookie
    request = Request(f"{base_url.rstrip('/')}/{endpoint.lstrip('/')}{query}", headers=headers, method="GET")
    opener = build_opener(ProxyHandler({}), _NoRedirect())
    try:
        with opener.open(request, timeout=10) as response:
            body = response.read(MAX_BODY + 1)
    except HTTPError as exc:
        raise LoginFailure(_request_error_code(endpoint, transient=exc.code >= 500 or exc.code == 429)) from None
    except (URLError, TimeoutError, OSError):
        raise LoginFailure(_request_error_code(endpoint, transient=True)) from None
    if len(body) > MAX_BODY:
        raise LoginFailure(_request_error_code(endpoint, transient=False))
    try:
        payload = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise LoginFailure(_request_error_code(endpoint, transient=False)) from None
    if not isinstance(payload, dict):
        raise LoginFailure(_request_error_code(endpoint, transient=False))
    return payload


def _request_error_code(endpoint: str, *, transient: bool) -> str:
    if endpoint == "login/qr/check":
        return "qr_check_unavailable" if transient else "qr_check_failed"
    if endpoint == "login/status":
        return "account_validation_unavailable" if transient else "account_validation_failed"
    return "qr_request_failed"


def _response_data(payload: dict) -> dict:
    data = payload.get("data")
    if payload.get("code") not in {None, 200} or not isinstance(data, dict):
        raise LoginFailure("qr_request_failed")
    return data


def _make_qr(qr_url: str, target: Path) -> None:
    import qrcode

    qr = qrcode.QRCode(border=1, box_size=8)
    qr.add_data(qr_url)
    qr.make(fit=True)
    image = qr.make_image(fill_color="black", back_color="white")
    target.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile("wb", prefix=f".{target.name}.", suffix=".tmp.png", dir=target.parent, delete=False) as output:
            temp_path = Path(output.name)
        image.save(temp_path, format="PNG")
        data = temp_path.read_bytes()
        if not data.startswith(PNG_SIGNATURE) or len(data) > 2 * 1024 * 1024:
            raise RuntimeError("QR image invalid")
        os.replace(temp_path, target)
        temp_path = None
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def _atomic_credentials(path: Path, profile: str, cookie: str) -> None:
    normalized = normalize_cookie(cookie)
    if not normalized:
        raise LoginFailure("account_validation_failed")
    payload = json.dumps({"profile": profile, "cookie": normalized, "saved_at": int(time.time())}, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    _atomic_write(path, payload)


def run(base_url: str, profile: str, job_dir: Path, max_wait_seconds: int = 180) -> int:
    parts = urlsplit(base_url)
    if parts.scheme != "http" or parts.hostname not in {"127.0.0.1", "localhost", "::1"} or parts.port is None:
        raise LoginFailure("qr_request_failed")
    if not PROFILE_RE.fullmatch(profile):
        raise LoginFailure("internal_error")
    root = job_dir.resolve(strict=True)
    status_path = root / "status.json"
    qr_path = root / "qr.png"
    credential_path = root.parent.parent / "credentials" / f"{profile}.json"
    _publish(status_path, "starting", "正在向网易云音乐申请二维码…")
    timestamp = int(time.time() * 1000)
    key_data = _response_data(_request(base_url, "login/qr/key", {"timestamp": timestamp}))
    key = key_data.get("unikey") or key_data.get("key")
    if not isinstance(key, str) or not key or len(key) > 512:
        raise LoginFailure("qr_request_failed")
    create_data = _response_data(_request(base_url, "login/qr/create", {"key": key, "qrimg": "false", "timestamp": timestamp}))
    qr_url = create_data.get("qrurl")
    if not isinstance(qr_url, str) or not qr_url.startswith("https://") or len(qr_url) > 4096:
        raise LoginFailure("qr_request_failed")
    _make_qr(qr_url, qr_path)
    _publish(status_path, "waiting", "请用网易云音乐 App 扫描二维码。")
    start = time.monotonic()
    last_state = "waiting"
    while time.monotonic() - start < max_wait_seconds:
        time.sleep(2)
        payload = _request(base_url, "login/qr/check", {"key": key, "timestamp": int(time.time() * 1000)})
        code = payload.get("code")
        if code == 800:
            _publish(status_path, "expired", "二维码已过期，请重新生成。")
            return 2
        if code == 801:
            if last_state != "waiting":
                last_state = "waiting"
                _publish(status_path, "waiting", "请用网易云音乐 App 扫描二维码。")
            continue
        if code == 802:
            if last_state != "scanned":
                last_state = "scanned"
                _publish(status_path, "scanned", "已扫码，请在网易云音乐 App 确认登录。")
            continue
        if code == 803:
            data = payload.get("data") if isinstance(payload.get("data"), dict) else {}
            cookie = normalize_cookie(payload.get("cookie") or data.get("cookie") or "")
            if not cookie:
                raise LoginFailure("account_validation_failed")
            status = _request(base_url, "login/status", {"timestamp": int(time.time() * 1000)}, cookie=cookie)
            status_data = status.get("data") if isinstance(status.get("data"), dict) else {}
            account = status_data.get("account")
            if status.get("code") not in {None, 200} or status_data.get("code") != 200 or not isinstance(account, dict) or not account.get("id"):
                raise LoginFailure("account_validation_failed")
            _atomic_credentials(credential_path, profile, cookie)
            _publish(status_path, "success", "网易云音乐登录成功，凭据已安全保存。")
            return 0
        raise LoginFailure("qr_check_failed")
    _publish(status_path, "expired", "二维码登录超时，请重新生成。")
    return 2


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--job-dir", required=True)
    args = parser.parse_args()
    try:
        return run(args.base_url, args.profile, Path(args.job_dir))
    except LoginFailure as error:
        try:
            root = Path(args.job_dir).resolve(strict=True)
            _publish(root / "status.json", "error", _ERROR_MESSAGES[error.code], error_code=error.code)
        except Exception:
            pass
        return 1
    except BaseException:
        try:
            root = Path(args.job_dir).resolve(strict=True)
            _publish(root / "status.json", "error", _ERROR_MESSAGES["internal_error"], error_code="internal_error")
        except Exception:
            pass
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
