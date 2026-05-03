#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"
PYTHON_BIN="python3.12"
RECREATE=0

usage() {
  cat <<'EOF'
Usage: scripts/setup_env.sh [options]

Create (or recreate) a project virtual environment and install pinned dependencies.

Options:
  --recreate           Remove the existing venv first, then create a clean one
  --venv-dir PATH      Virtual environment directory (default: .venv)
  --python BIN         Python executable to use for venv creation (default: python3.12)
  -h, --help           Show this help message

Examples:
  scripts/setup_env.sh
  scripts/setup_env.sh --recreate
  scripts/setup_env.sh --python python3.12 --venv-dir .venv312 --recreate
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --recreate)
      RECREATE=1
      shift
      ;;
    --venv-dir)
      VENV_DIR="$2"
      shift 2
      ;;
    --python)
      PYTHON_BIN="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ ! -f "${ROOT_DIR}/requirements.txt" ]]; then
  echo "requirements.txt not found in ${ROOT_DIR}" >&2
  exit 1
fi

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Python executable not found: ${PYTHON_BIN}" >&2
  exit 1
fi

if [[ "${RECREATE}" -eq 1 && -d "${VENV_DIR}" ]]; then
  echo "Removing existing virtual environment: ${VENV_DIR}"
  rm -rf "${VENV_DIR}"
fi

if [[ ! -d "${VENV_DIR}" ]]; then
  echo "Creating virtual environment with ${PYTHON_BIN}: ${VENV_DIR}"
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
else
  echo "Using existing virtual environment: ${VENV_DIR}"
fi

"${VENV_DIR}/bin/python" -m pip install --upgrade pip setuptools wheel
"${VENV_DIR}/bin/python" -m pip install -r "${ROOT_DIR}/requirements.txt"

echo
echo "Environment is ready."
echo "Activate it with: source ${VENV_DIR}/bin/activate"
echo "Or run directly with: ${VENV_DIR}/bin/python process_videos.py"
