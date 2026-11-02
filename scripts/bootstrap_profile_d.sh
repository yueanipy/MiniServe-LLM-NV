#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
profile_data_root="${XDG_DATA_HOME:-${HOME}/.local/share}/miniserve-llm-nv/profile-d"
venv_path="${profile_data_root}/venv"
cuda_home="${MINISERVE_CUDA_HOME:-/usr/local/cuda-13.0}"

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "BLOCKED: Profile D bootstrap must run inside Linux/WSL2." >&2
  exit 2
fi
if [[ ! -x "${cuda_home}/bin/nvcc" ]]; then
  echo "BLOCKED: CUDA 13.0 toolkit is missing at ${cuda_home}." >&2
  exit 2
fi

export CUDA_HOME="${cuda_home}"
export PATH="${CUDA_HOME}/bin:${PATH}"
for command_name in python3 nvcc gcc g++ nvidia-smi cuobjdump; do
  if ! command -v "${command_name}" >/dev/null 2>&1; then
    echo "BLOCKED: required Linux command '${command_name}' is missing." >&2
    exit 2
  fi
done
if ! nvcc --version | grep -Fq "release 13.0"; then
  echo "BLOCKED: Profile D requires the locked CUDA 13.0 compiler." >&2
  nvcc --version >&2
  exit 2
fi

mkdir -p "${profile_data_root}"
python3 -m venv "${venv_path}"
"${venv_path}/bin/python" -m pip install --upgrade "pip==26.2.1"
"${venv_path}/bin/python" -m pip install -r "${repo_root}/requirements/profile-d.txt"
"${venv_path}/bin/python" -m pip install "torch==2.13.0" \
  --index-url https://download.pytorch.org/whl/cu130

echo "Profile D virtual environment prepared at ${venv_path}."
echo "Next: source scripts/activate_profile_d.sh && bash scripts/build_native.sh D"
