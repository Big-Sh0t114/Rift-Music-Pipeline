"""Download one explicitly requested, pinned runtime asset with resume support."""

from __future__ import annotations

import argparse
import hashlib
import os
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


MAX_ASSET_BYTES = 8 * 1024 * 1024 * 1024
CHUNK_SIZE = 1024 * 1024


class _HttpsRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, newurl):
        if urllib.parse.urlsplit(newurl).scheme != "https":
            raise urllib.error.HTTPError(newurl, code, "HTTPS redirect required", headers, fp)
        return super().redirect_request(request, fp, code, message, headers, newurl)


def _digest(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as source:
        while chunk := source.read(CHUNK_SIZE):
            size += len(chunk)
            digest.update(chunk)
    return digest.hexdigest(), size


def download(url: str, target: Path, expected_sha: str, expected_size: int | None) -> None:
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != "https" or not parsed.hostname or len(expected_sha) != 64:
        raise ValueError("invalid asset pin")
    target = target.expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + ".partial")
    if target.is_file():
        digest, size = _digest(target)
        if digest == expected_sha.casefold() and (expected_size is None or size == expected_size):
            return

    offset = part.stat().st_size if part.is_file() else 0
    if offset > MAX_ASSET_BYTES or (expected_size is not None and offset > expected_size):
        part.unlink(missing_ok=True)
        offset = 0
    elif offset and expected_size is not None and offset == expected_size:
        partial_digest, partial_size = _digest(part)
        if partial_size == expected_size and partial_digest == expected_sha.casefold():
            os.replace(part, target)
            return
        part.unlink(missing_ok=True)
        offset = 0
    request = urllib.request.Request(url, headers={"User-Agent": "NachoBot-mus-library-runtime", "Accept-Encoding": "identity"})
    if offset:
        request.add_header("Range", f"bytes={offset}-")
    opener = urllib.request.build_opener(_HttpsRedirect())
    try:
        response = opener.open(request, timeout=45)
    except urllib.error.HTTPError as exc:
        if exc.code != 416 or offset <= 0:
            raise
        part.unlink(missing_ok=True)
        offset = 0
        request = urllib.request.Request(url, headers={"User-Agent": "NachoBot-mus-library-runtime", "Accept-Encoding": "identity"})
        response = opener.open(request, timeout=45)
    with response:
        if urllib.parse.urlsplit(response.geturl()).scheme != "https":
            raise ValueError("insecure final URL")
        status = getattr(response, "status", 200)
        if offset and status == 206:
            mode = "ab"
        elif status == 200:
            offset = 0
            mode = "wb"
        else:
            raise ValueError("unexpected response")
        digest = hashlib.sha256()
        if offset:
            with part.open("rb") as previous:
                while chunk := previous.read(CHUNK_SIZE):
                    digest.update(chunk)
        written = offset
        with part.open(mode) as output:
            while chunk := response.read(CHUNK_SIZE):
                written += len(chunk)
                if written > MAX_ASSET_BYTES or (expected_size is not None and written > expected_size):
                    raise ValueError("asset exceeded pinned size")
                output.write(chunk)
                digest.update(chunk)
            output.flush()
            os.fsync(output.fileno())
    if expected_size is not None and written != expected_size:
        raise ValueError("asset size mismatch")
    if digest.hexdigest().casefold() != expected_sha.casefold():
        part.unlink(missing_ok=True)
        raise ValueError("asset checksum mismatch")
    os.replace(part, target)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--url", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--size", type=int)
    args = parser.parse_args(argv)
    try:
        download(args.url, Path(args.target), args.sha256, args.size)
        return 0
    except (OSError, ValueError, urllib.error.URLError):
        # Error text may contain local paths or signed CDN URLs; keep the public
        # installer surface to a stable exit code.
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
