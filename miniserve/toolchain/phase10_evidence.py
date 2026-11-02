"""记录 Phase 10 实际解析的 Web tuple，避免把 requirements 当作安装事实。"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from datetime import datetime, timezone
from pathlib import Path


PACKAGES = (
    "fastapi",
    "pydantic",
    "pydantic-core",
    "uvicorn",
    "httpx",
    "httpcore",
    "starlette",
    "anyio",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = {
        "schema_version": 1,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "packages": {name: importlib.metadata.version(name) for name in PACKAGES},
        "contract": {
            "endpoint": "/v1/chat/completions",
            "transport": ["json", "sse"],
            "default_bind": "127.0.0.1",
            "remote_auth": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
