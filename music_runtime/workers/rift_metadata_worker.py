"""Read only RIFT speaker labels from an explicit local checkpoint, CPU/meta only."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


MAX_RESPONSE = 128 * 1024


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    try:
        import torch

        checkpoint = Path(args.checkpoint).resolve(strict=True)
        if not checkpoint.is_file() or checkpoint.suffix.casefold() not in {".ckpt", ".pt", ".pth"}:
            raise ValueError("invalid checkpoint")
        # This helper reads a user-selected local checkpoint only. Meta-device
        # storage avoids allocating model weights or touching CUDA; never
        # import infer.py or pretrained modules for speaker discovery.
        data = torch.load(checkpoint, map_location="meta", weights_only=False, mmap=True)
        hyper = data.get("hyper_parameters") if isinstance(data, dict) else None
        cfg = hyper.get("cfg") if isinstance(hyper, dict) else None
        spk2idx = cfg.get("spk2idx") if isinstance(cfg, dict) else None
        if not isinstance(spk2idx, dict) or not spk2idx or len(spk2idx) > 4096:
            raise ValueError("speaker map missing")
        speakers = []
        for name, index in spk2idx.items():
            if not isinstance(name, str) or len(name) > 128 or type(index) is not int or index < 0:
                raise ValueError("speaker map invalid")
            speakers.append({"id": name, "name": name, "index": index})
        speakers.sort(key=lambda item: (item["index"], item["id"]))
        payload = json.dumps({"ok": True, "speakers": speakers[:2048]}, ensure_ascii=False, separators=(",", ":"))
        if len(payload.encode("utf-8")) > MAX_RESPONSE:
            raise ValueError("speaker map too large")
        sys.stdout.write(payload)
        return 0
    except Exception:
        sys.stdout.write('{"ok":false,"speakers":[]}')
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
