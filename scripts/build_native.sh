#!/usr/bin/env bash
# MiniServe 最终项目的干净 native wheel 构建入口。
set -euo pipefail

profile="${1:-L}"
case "${profile}" in
  L) architecture="75"; profile_dir="profile-l" ;;
  D) architecture="120"; profile_dir="profile-d" ;;
  *) echo "Usage: bash scripts/build_native.sh L|D" >&2; exit 2 ;;
esac

if [[ "$(uname -s)" != "Linux" || -z "${VIRTUAL_ENV:-}" ]]; then
  echo "BLOCKED: run inside Linux/WSL with the MiniServe profile venv activated." >&2
  exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cache_root="${XDG_CACHE_HOME:-${HOME}/.cache}/miniserve-llm-nv/final/${profile_dir}"
build_dir="${cache_root}/build"
wheel_cache="${cache_root}/wheel"
expected="${cache_root}/build"
expected_wheel_cache="${cache_root}/wheel"
if [[ "${build_dir}" != "${expected}" || "${wheel_cache}" != "${expected_wheel_cache}" ]]; then
  echo "Refusing to clean unexpected build directory: ${build_dir}" >&2
  exit 2
fi

cd "${repo_root}"
rm -rf "${build_dir}"
rm -rf "${wheel_cache}"
mkdir -p "${wheel_cache}"
SKBUILD_BUILD_DIR="${build_dir}" \
  python -m pip wheel --verbose --no-build-isolation --no-deps \
  --wheel-dir "${wheel_cache}" \
  -C"cmake.define.CMAKE_CUDA_ARCHITECTURES=${architecture}" .

mapfile -t wheels < <(find "${wheel_cache}" -maxdepth 1 -type f -name 'miniserve_llm_nv-*.whl' -print)
if [[ "${#wheels[@]}" -ne 1 ]]; then
  echo "Expected exactly one freshly built MiniServe wheel, found ${#wheels[@]}." >&2
  exit 2
fi
python -m pip install --force-reinstall --no-deps "${wheels[0]}"

# 将本轮真正安装的 wheel 与摘要作为最终证据持久化；文件仅约 0.2 MB。
evidence_wheel_dir="${repo_root}/evidence/final/${profile_dir}/generated/wheel"
mkdir -p "${evidence_wheel_dir}"
cp -f "${wheels[0]}" "${evidence_wheel_dir}/"
sha256sum "${evidence_wheel_dir}/$(basename "${wheels[0]}")" \
  > "${evidence_wheel_dir}/SHA256SUMS"

echo "Installed and persisted clean MiniServe native wheel for ${profile_dir} / sm_${architecture}."
