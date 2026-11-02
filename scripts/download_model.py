#!/usr/bin/env python3
"""下载受支持模型、固定 commit，并生成带逐文件 SHA-256 的 model lock。"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path


DOWNLOAD_PATTERNS = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "vocab.json",
    "merges.txt",
    "*.safetensors",
    "*.safetensors.index.json",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--revision", default="main")
    parser.add_argument(
        "--lock-path",
        type=Path,
        default=Path("model-locks/qwen3-0.6b-profile-l.json"),
    )
    return parser.parse_args()


def main() -> int:
    from huggingface_hub import HfApi, snapshot_download

    args = parse_args()
    resolved_revision = HfApi().model_info(
        args.repo_id, revision=args.revision
    ).sha
    snapshot = Path(
        snapshot_download(
            repo_id=args.repo_id,
            revision=resolved_revision,
            allow_patterns=list(DOWNLOAD_PATTERNS),
        )
    ).resolve()
    files = sorted(path for path in snapshot.rglob("*") if path.is_file())
    required = (snapshot / "config.json", snapshot / "tokenizer.json")
    if not all(path.is_file() for path in required):
        raise RuntimeError("Downloaded snapshot is missing config.json or tokenizer.json.")
    if not any(path.name.endswith(".safetensors") for path in files):
        raise RuntimeError("Downloaded snapshot has no safetensors weights.")

    tokenizer_config = json.loads(
        (snapshot / "tokenizer_config.json").read_text(encoding="utf-8")
    )
    if (snapshot / "chat_template.jinja").is_file():
        chat_template_source = "chat_template.jinja"
    elif "chat_template" in tokenizer_config:
        chat_template_source = "tokenizer_config.json#chat_template"
    else:
        chat_template_source = None
    lock = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "model_id": args.repo_id,
        "requested_revision": args.revision,
        "resolved_revision": resolved_revision,
        "local_snapshot": str(snapshot),
        "chat_template_source": chat_template_source,
        "files": {
            str(path.relative_to(snapshot)): {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in files
        },
    }
    args.lock_path.parent.mkdir(parents=True, exist_ok=True)
    args.lock_path.write_text(
        json.dumps(lock, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "model_id": args.repo_id,
                "snapshot": str(snapshot),
                "resolved_revision": resolved_revision,
                "lock_path": str(args.lock_path),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
