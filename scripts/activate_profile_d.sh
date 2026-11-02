#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "This script must be sourced so the virtual environment remains active." >&2
  echo "Run: source scripts/activate_profile_d.sh" >&2
  exit 2
fi

profile_data_root="${XDG_DATA_HOME:-${HOME}/.local/share}/miniserve-llm-nv/profile-d"
venv_path="${profile_data_root}/venv"
cuda_home="${MINISERVE_CUDA_HOME:-/usr/local/cuda-13.0}"

if [[ ! -f "${venv_path}/bin/activate" ]]; then
  echo "Profile D virtual environment is missing." >&2
  echo "Run first: bash scripts/bootstrap_profile_d.sh" >&2
  return 2
fi
if [[ ! -x "${cuda_home}/bin/nvcc" ]]; then
  echo "Profile D CUDA compiler is missing: ${cuda_home}/bin/nvcc" >&2
  return 2
fi

export CUDA_HOME="${cuda_home}"
export PATH="${CUDA_HOME}/bin:${PATH}"
source "${venv_path}/bin/activate"
echo "Activated Profile D environment: ${venv_path} (CUDA_HOME=${CUDA_HOME})"
