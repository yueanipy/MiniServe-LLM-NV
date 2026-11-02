"""验证 Linux 动态库依赖与目标 GPU 原生代码是否真实进入共享库。

成功 import 只能说明当前进程碰巧能加载；本工具额外用 ``ldd`` 和 ``cuobjdump``
证明 clean runtime 依赖可解析，并且包含当前 Profile 要求的 cubin。
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from miniserve._native import find_extension, load_native_ops
from miniserve.toolchain.probe import PROFILE_CAPABILITIES, write_json


def _run(command: list[str]) -> dict[str, Any]:
    """运行一个有 30 秒上限的只读二进制检查，并保留 stdout/stderr。"""

    # check=False 让外部工具失败成为结构化 Gate 数据，而不是丢失上下文的异常。
    completed = subprocess.run(command, check=False, capture_output=True, text=True, timeout=30)
    output = "\n".join(
        item.strip() for item in (completed.stdout, completed.stderr) if item.strip()
    )
    return {"command": command, "returncode": completed.returncode, "output": output}


def verify(profile: str) -> dict[str, Any]:
    """对指定 Profile 的唯一扩展执行加载、依赖和架构三层验证。"""

    normalized = profile.upper()
    expected = PROFILE_CAPABILITIES[normalized]
    # capability [7, 5] 对应 cuobjdump 名称 sm_75；[12, 0] 对应 sm_120。
    expected_arch = f"sm_{expected[0]}{expected[1]}"

    library_path = find_extension()
    # 先真实加载一次，验证静态 Dispatcher 注册不会在 dlopen 阶段失败。
    load_native_ops()

    # shutil.which 记录当前激活环境实际选择的工具，而不是假设固定安装位置。
    ldd_path = shutil.which("ldd")
    cuobjdump_path = shutil.which("cuobjdump")
    ldd = _run([ldd_path, str(library_path)]) if ldd_path else {"returncode": None, "output": "ldd not found"}
    cubins = _run([cuobjdump_path, "--list-elf", str(library_path)]) if cuobjdump_path else {"returncode": None, "output": "cuobjdump not found"}

    # 每个布尔项回答一个不同问题，不能用“文件存在”替代其他检查。
    checks = {
        "library_loaded": True,
        "ldd_succeeded": ldd["returncode"] == 0,
        "dependencies_resolved": ldd["returncode"] == 0 and "not found" not in ldd["output"].lower(),
        "cuobjdump_succeeded": cubins["returncode"] == 0,
        "native_arch_present": expected_arch in cubins["output"],
    }
    return {
        "schema_version": 1,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "profile": normalized,
        "library_path": str(library_path),
        "expected_arch": expected_arch,
        "ldd": ldd,
        "cuobjdump": cubins,
        "checks": checks,
        "passed": all(checks.values()),
    }


def main(argv: list[str] | None = None) -> int:
    """CLI 入口：可选写 JSON，并用退出码表达二进制 Gate。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("L", "D", "l", "d"), required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    report = verify(args.profile)
    if args.output is not None:
        write_json(report, args.output)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
