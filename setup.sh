#!/usr/bin/env bash
# Bootstrap system tools and build the Swift-only DeepSeek harness.
set -euo pipefail

REFERENCE_MODEL_REPO="${MLXFAST_REFERENCE_MODEL_REPO:-mlx-community/DeepSeek-V4-Flash-4bit}"
REFERENCE_MIN_FREE_GIB="${MLXFAST_REFERENCE_MIN_FREE_GIB:-170}"
REFERENCE_LFS_INCLUDE="${MLXFAST_REFERENCE_LFS_INCLUDE:-*.json,model*.safetensors,tokenizer*,*.tiktoken,tiktoken.model,*.txt,chat_template.jinja}"

load_homebrew_shellenv() {
  local candidate
  local candidates=()

  if [[ -n "${HOMEBREW_PREFIX:-}" ]]; then
    candidates+=("${HOMEBREW_PREFIX}/bin/brew")
  fi
  candidates+=(
    "/opt/homebrew/bin/brew"
    "/usr/local/bin/brew"
    "${HOME}/.linuxbrew/bin/brew"
  )

  for candidate in "${candidates[@]}"; do
    if [[ -x "${candidate}" ]]; then
      eval "$("${candidate}" shellenv)"
      return 0
    fi
  done

  return 1
}

ensure_homebrew() {
  if command -v brew >/dev/null 2>&1; then
    return 0
  fi

  if load_homebrew_shellenv && command -v brew >/dev/null 2>&1; then
    return 0
  fi

  if [[ "${MLXFAST_SKIP_HOMEBREW_INSTALL:-0}" == "1" ]]; then
    echo "setup.sh: Homebrew is not installed and MLXFAST_SKIP_HOMEBREW_INSTALL=1" >&2
    return 1
  fi

  if [[ "$(uname -s)" != "Darwin" ]]; then
    echo "setup.sh: automatic Homebrew installation is only supported on macOS" >&2
    return 1
  fi

  if ! command -v curl >/dev/null 2>&1; then
    echo "setup.sh: curl is required to install Homebrew" >&2
    return 1
  fi

  echo "setup.sh: Homebrew not found; installing Homebrew"
  NONINTERACTIVE=1 /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

  if ! load_homebrew_shellenv || ! command -v brew >/dev/null 2>&1; then
    echo "setup.sh: Homebrew installation finished, but brew is still not on PATH" >&2
    echo "setup.sh: open a new shell or run Homebrew's shellenv command, then retry" >&2
    return 1
  fi
}

ensure_mactop() {
  if [[ "${MLXFAST_SKIP_MACTOP_INSTALL:-0}" == "1" ]]; then
    echo "setup.sh: skipping mactop install"
    return 0
  fi

  if [[ -n "${MLXFAST_MACTOP_BIN:-}" ]]; then
    if [[ -x "${MLXFAST_MACTOP_BIN}" ]]; then
      echo "setup.sh: using mactop at ${MLXFAST_MACTOP_BIN}"
      return 0
    fi
    echo "setup.sh: MLXFAST_MACTOP_BIN is set but not executable: ${MLXFAST_MACTOP_BIN}" >&2
    return 1
  fi

  if [[ "$(uname -s)" != "Darwin" ]]; then
    echo "setup.sh: skipping mactop install; mactop is only available on macOS"
    return 0
  fi

  if command -v mactop >/dev/null 2>&1 || [[ -x "/opt/homebrew/bin/mactop" || -x "/usr/local/bin/mactop" ]]; then
    return 0
  fi

  ensure_homebrew
  echo "setup.sh: installing mactop with Homebrew"
  brew install mactop

  if ! command -v mactop >/dev/null 2>&1 && [[ ! -x "/opt/homebrew/bin/mactop" && ! -x "/usr/local/bin/mactop" ]]; then
    echo "setup.sh: mactop installation finished, but the mactop binary was not found" >&2
    return 1
  fi
}

ensure_git_lfs() {
  if command -v git-lfs >/dev/null 2>&1 && git lfs version >/dev/null 2>&1; then
    return 0
  fi

  ensure_homebrew
  echo "setup.sh: installing git-lfs with Homebrew"
  brew install git-lfs

  if ! command -v git-lfs >/dev/null 2>&1 || ! git lfs version >/dev/null 2>&1; then
    echo "setup.sh: git-lfs installation finished, but git lfs is not usable" >&2
    return 1
  fi
}

ensure_swift_toolchain() {
  if [[ "$(uname -s)" != "Darwin" ]]; then
    echo "setup.sh: this Swift harness targets macOS on Apple Silicon" >&2
    exit 1
  fi

  if ! command -v swift >/dev/null 2>&1; then
    echo "setup.sh: swift was not found; install Xcode command line tools" >&2
    exit 1
  fi

  if ! command -v xcodebuild >/dev/null 2>&1; then
    echo "setup.sh: xcodebuild was not found; install Xcode" >&2
    exit 1
  fi
}

ensure_reference_space() {
  local directory="$1"
  local available_kib
  local required_kib

  if ! [[ "${REFERENCE_MIN_FREE_GIB}" =~ ^[0-9]+$ ]]; then
    echo "setup.sh: MLXFAST_REFERENCE_MIN_FREE_GIB must be an integer" >&2
    return 1
  fi

  available_kib="$(df -Pk "${directory}" | awk 'NR == 2 {print $4}')"
  required_kib=$((REFERENCE_MIN_FREE_GIB * 1024 * 1024))
  if [[ -z "${available_kib}" || "${available_kib}" -lt "${required_kib}" ]]; then
    cat >&2 <<EOF
setup.sh: not enough free disk space for ${REFERENCE_MODEL_REPO}.

Need at least ${REFERENCE_MIN_FREE_GIB} GiB free under ${directory}; available is $((available_kib / 1024 / 1024)) GiB.
Set MLXFAST_REFERENCE_DIR to a larger SSD, or set MLXFAST_SKIP_WEIGHTS_DOWNLOAD=1
and place/mount the checkpoint manually.

EOF
    return 1
  fi
}

download_reference_weights() {
  local reference_dir="$1"
  local parent_dir
  local partial_dir
  local repo_url

  if [[ -f "${reference_dir}/config.json" ]]; then
    echo "setup.sh: reference weights already present at ${reference_dir}"
    return 0
  fi

  if [[ -e "${reference_dir}" ]]; then
    cat >&2 <<EOF
setup.sh: ${reference_dir} exists but does not contain config.json.

Move it aside or set MLXFAST_REFERENCE_DIR to a complete checkpoint directory.

EOF
    return 1
  fi

  parent_dir="$(dirname "${reference_dir}")"
  partial_dir="${reference_dir}.partial"
  repo_url="${MLXFAST_REFERENCE_REPO_URL:-https://huggingface.co/${REFERENCE_MODEL_REPO}}"
  mkdir -p "${parent_dir}"

  ensure_reference_space "${parent_dir}"
  ensure_git_lfs

  if [[ -d "${partial_dir}/.git" ]]; then
    echo "setup.sh: resuming reference weight download in ${partial_dir}"
  elif [[ -e "${partial_dir}" ]]; then
    echo "setup.sh: partial download path exists but is not a git repo: ${partial_dir}" >&2
    return 1
  else
    echo "setup.sh: cloning ${REFERENCE_MODEL_REPO} metadata"
    GIT_LFS_SKIP_SMUDGE=1 git clone "${repo_url}" "${partial_dir}"
  fi

  echo "setup.sh: downloading reference safetensors with git-lfs"
  git -C "${partial_dir}" lfs install --local
  git -C "${partial_dir}" lfs pull --include "${REFERENCE_LFS_INCLUDE}" --exclude ""

  if [[ ! -f "${partial_dir}/config.json" ]]; then
    echo "setup.sh: downloaded checkpoint is missing config.json" >&2
    return 1
  fi

  mv "${partial_dir}" "${reference_dir}"
  echo "setup.sh: downloaded reference weights to ${reference_dir}"
}

ensure_mactop

ensure_swift_toolchain

echo "setup.sh: building Swift harness"
mkdir -p .build/clang-module-cache
export CLANG_MODULE_CACHE_PATH="${CLANG_MODULE_CACHE_PATH:-${PWD}/.build/clang-module-cache}"
swift build -c release

if [[ "${MLXFAST_SKIP_MLX_METALLIB:-0}" == "1" ]]; then
  echo "setup.sh: skipping mlx.metallib build"
else
  echo "setup.sh: building mlx.metallib for MLX Swift runtime"
  tools/build-mlx-metallib.sh
fi

REFERENCE_DIR="${MLXFAST_REFERENCE_DIR:-reference_weights/DeepSeek-V4-Flash-4bit}"

if [[ "${MLXFAST_SKIP_WEIGHTS_DOWNLOAD:-0}" == "1" || "${SKIP_MODEL_DOWNLOAD:-0}" == "1" ]]; then
  echo "setup.sh: skipping reference weight download"
  exit 0
fi

download_reference_weights "${REFERENCE_DIR}"
