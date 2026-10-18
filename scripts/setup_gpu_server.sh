#!/usr/bin/env bash
# Bootstrap a fresh Linux GPU server for VLM-DENTAL SFT (patch 0039).
#
#   git clone https://github.com/rezaxr14/VLM-DENTAL.git && cd VLM-DENTAL
#   bash scripts/setup_gpu_server.sh                 # idempotent: run it again after any stop, it continues
#   bash scripts/setup_gpu_server.sh --report-only   # machine report + the decisions it would take, installs nothing
#
# After it finishes, open notebooks/VLM_Dental_Colab_SFT.ipynb with the kernel "VLM-DENTAL (.venv)".
# The decisions (torch wheel index, attention backend, .env rules) live in dental_agent/utils/gpu_setup.py and are unit-tested
# without a GPU; this file only runs commands. Details: docs/RUNBOOK_SFT_GRPO.md, "A0. Fresh GPU server".
#
# What it touches, and nothing else: <repo>/.venv, <repo>/.env, <repo>/data, ~/.cache/vlm-dental, ~/.local/bin/uv (+ uv's own
# folders), ~/.local/share/jupyter/kernels/vlm-dental, and apt packages ONLY when sudo is available and you allow it.
# What it never does: commit or push, delete, kill a process, change a training setting (precision, sequence length, LoRA,
# hyperparameters, notebook cells), print a token value, or run tune2fs.
set -Eeuo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HELPER="$REPO/dental_agent/utils/gpu_setup.py"

VENV="$REPO/.venv"
STATE_DIR="${VLM_STATE_DIR:-$HOME/.cache/vlm-dental}"
WHEELHOUSE="${VLM_WHEELHOUSE:-$STATE_DIR/wheelhouse}"
STAGE="dentex_alone"
TRACK="with_tools"
TORCH_VER="" TV_VER="" CUDA_INDEX="" FA_WHEEL="" MIN_FREE=""
NO_FA=0 NO_CC1D=0 NO_KERNEL=0 NO_SUDO=0 DRY=0 REPORT_ONLY=0 SKIP_DATA=0 SKIP_MODEL=0 SKIP_PREFLIGHT=0
PREFLIGHT_EXTRA=()
HARD_FAIL=0
CUR_STEP="start"

usage() {
  cat <<'EOF'
Usage: bash scripts/setup_gpu_server.sh [options] [-- <extra train_sft.py flags for the final preflight>]

  --report-only          print the machine report and the decisions (torch build, attention plan); install nothing
  --dry-run              also list every command it would run (needs no venv; nothing is changed)
  --no-sudo              never call sudo; print the apt command for missing system packages instead
  --venv DIR             virtual environment (default: <repo>/.venv; an existing working one is reused, never deleted)
  --wheelhouse DIR       where torch wheels are kept so a stalled download resumes (default: ~/.cache/vlm-dental/wheelhouse)
  --torch X --torchvision Y   override the versions read from the commented lines of requirements.txt (give both)
  --cuda-index cuNNN     override the torch CUDA build chosen from driver + compute capability
  --flash-attn-wheel URL install this flash-attn wheel instead of looking for an official one (a third-party binary is your call)
  --no-flash-attn        never install flash-attn (train_sft.py then uses PyTorch SDPA)
  --no-causal-conv1d     skip the optional causal-conv1d kernel
  --no-kernel            do not install ipykernel / register the Jupyter kernel "VLM-DENTAL (.venv)"
  --stage S --track T    which traces/images to fetch and to preflight (default: dentex_alone / with_tools)
  --skip-model | --skip-data | --skip-preflight
  --min-free-gb N        replace the built-in free-disk requirement at the repo (the figures are allowances, not measurements)
  -h, --help
EOF
}

say()  { printf '%s\n' "$*"; }
step() { CUR_STEP="$1 $2"; printf '\n[%s] %s\n' "$1" "$2"; }
warn() { printf 'WARNING: %s\n' "$*" >&2; }
die()  { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
run()  { if ((DRY)); then printf '  (dry-run) %s\n' "$*"; else "$@"; fi; }
need_arg() { [[ $# -ge 2 && -n "$2" ]] || die "$1 needs a value (see --help)"; }
on_err() { printf '\nStopped in step [%s] (line %s). Everything finished so far is kept; run the script again to continue.\n' "$CUR_STEP" "$1" >&2; }
trap 'on_err $LINENO' ERR
expect_status() { trap - ERR; set +e; }          # a command below may fail on purpose: its exit code is read, not fatal
end_expect()    { set -e; trap 'on_err $LINENO' ERR; }

while (($#)); do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --report-only) REPORT_ONLY=1 ;;
    --dry-run) DRY=1 ;;
    --no-sudo) NO_SUDO=1 ;;
    --no-flash-attn) NO_FA=1 ;;
    --no-causal-conv1d) NO_CC1D=1 ;;
    --no-kernel) NO_KERNEL=1 ;;
    --skip-model) SKIP_MODEL=1 ;;
    --skip-data) SKIP_DATA=1 ;;
    --skip-preflight) SKIP_PREFLIGHT=1 ;;
    --venv) need_arg "$@"; VENV="$(realpath -m "$2")"; shift ;;
    --wheelhouse) need_arg "$@"; WHEELHOUSE="$(realpath -m "$2")"; shift ;;
    --torch) need_arg "$@"; TORCH_VER="$2"; shift ;;
    --torchvision) need_arg "$@"; TV_VER="$2"; shift ;;
    --cuda-index) need_arg "$@"; CUDA_INDEX="$2"; shift ;;
    --flash-attn-wheel) need_arg "$@"; FA_WHEEL="$2"; shift ;;
    --stage) need_arg "$@"; STAGE="$2"; shift ;;
    --track) need_arg "$@"; TRACK="$2"; shift ;;
    --min-free-gb) need_arg "$@"; MIN_FREE="$2"; shift ;;
    --) shift; PREFLIGHT_EXTRA=("$@"); break ;;
    *) die "unknown option: $1 (see --help)" ;;
  esac
  shift
done
[[ -z "$TORCH_VER" && -z "$TV_VER" ]] || [[ -n "$TORCH_VER" && -n "$TV_VER" ]] || die "give --torch and --torchvision together"
[[ "$(uname -s)" == "Linux" ]] || die "this script is for Linux GPU servers"
command -v python3 >/dev/null 2>&1 || die "python3 is needed to run the machine report (sudo apt-get install -y python3)"
[[ -f "$HELPER" ]] || die "missing $HELPER (run the script from a full checkout)"

mkdir -p "$STATE_DIR"
STATE="$STATE_DIR/setup-state.json"
printf '{}' >"$STATE"
VENV_PY="$VENV/bin/python"
gs() { python3 "$HELPER" "$@"; }                 # system Python, standard library only
gv() { "$VENV_PY" "$HELPER" "$@"; }              # the venv's Python (torch and the project importable)
rec() { ((DRY)) || gs record --state "$STATE" "$1" "$2"; }          # a dry run records no "done" claims
miss() { ((DRY)) || gs append-missing --state "$STATE" "$1"; }

# --------------------------------------------------------------------------------------------------------------------
step 1 "Inspect the machine"
expect_status
gs report --repo "$REPO" --venv "$VENV" --wheelhouse "$WHEELHOUSE" --state "$STATE" ${MIN_FREE:+--min-free-gb "$MIN_FREE"}
rc=$?
end_expect
case $rc in
  0) ;;
  2) die "no usable NVIDIA GPU (see the report above)" ;;
  3) die "not enough free disk (see the report above)" ;;
  *) die "the machine report failed (exit $rc)" ;;
esac

# --------------------------------------------------------------------------------------------------------------------
step 2 "Choose the torch build and the attention plan (from compute capability, never from a card name)"
if [[ -z "$TORCH_VER" ]]; then
  TORCH_VER="$(gs pin torch --requirements "$REPO/requirements.txt")" || die "could not read the torch version from requirements.txt"
  TV_VER="$(gs pin torchvision --requirements "$REPO/requirements.txt")" || die "could not read the torchvision version from requirements.txt"
fi
plan_args=(--state "$STATE" --torch-version "$TORCH_VER")
[[ -n "$CUDA_INDEX" ]] && plan_args+=(--cuda-index "$CUDA_INDEX")
((NO_FA)) && plan_args+=(--no-flash-attn)
gs plan "${plan_args[@]}" || die "no safe torch build could be chosen (see the message above)"
IDX="$(gs get --state "$STATE" torch_index)"
PLAN_FA="$(gs get --state "$STATE" plan_flash_attn2)"
if ((REPORT_ONLY)); then
  say ""; say "--report-only: nothing was installed."
  gs summary --state "$STATE"
  exit 0
fi

# --------------------------------------------------------------------------------------------------------------------
step 3 "System packages (only what is missing)"
PKGS=(git curl ca-certificates build-essential git-lfs aria2 tmux)
missing=()
if command -v dpkg-query >/dev/null 2>&1; then
  for p in "${PKGS[@]}"; do
    dpkg-query -W -f='${Status}' "$p" 2>/dev/null | grep -q 'install ok installed' || missing+=("$p")
  done
else
  warn "dpkg-query not found (not Debian/Ubuntu): make sure these exist yourself: ${PKGS[*]}"
fi
if ((${#missing[@]})); then
  say "missing: ${missing[*]}"
  SUDO=()
  if ((EUID != 0)); then SUDO=(sudo); fi
  if ((NO_SUDO)) || { ((EUID != 0)) && ! command -v sudo >/dev/null 2>&1; }; then
    say "sudo is not used here. To install them yourself:  sudo apt-get update && sudo apt-get install -y ${missing[*]}"
    miss "system packages not installed: ${missing[*]} (git/curl are needed by later steps; build-essential only to compile kernels)"
  else
    ((EUID != 0)) && say "sudo is needed ONLY for:  apt-get update && apt-get install -y ${missing[*]}"
    run "${SUDO[@]+"${SUDO[@]}"}" apt-get update
    run "${SUDO[@]+"${SUDO[@]}"}" env DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends "${missing[@]}"
  fi
else
  say "all present: ${PKGS[*]}"
fi

# --------------------------------------------------------------------------------------------------------------------
step 4 "Python 3.12 virtual environment with uv"
if ! command -v uv >/dev/null 2>&1 && [[ -x "$HOME/.local/bin/uv" ]]; then export PATH="$HOME/.local/bin:$PATH"; fi
if ! command -v uv >/dev/null 2>&1; then
  say "uv not found: installing it with Astral's official installer (curl | sh) into ~/.local/bin; shell startup files stay untouched."
  run env UV_NO_MODIFY_PATH=1 sh -c 'curl -LsSf https://astral.sh/uv/install.sh | sh'
  export PATH="$HOME/.local/bin:$PATH"
fi
venv_usable() {
  [[ -x "$VENV_PY" ]] \
    && "$VENV_PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null \
    && "$VENV_PY" -m pip --version >/dev/null 2>&1
}
if venv_usable; then
  say "using the existing $VENV ($("$VENV_PY" -V 2>&1))"
elif [[ -e "$VENV" ]]; then
  die "$VENV exists but is not a usable virtual environment. I never delete anything: move it away (mv '$VENV' '$VENV.old') and run again."
else
  say "Ubuntu 22.04 has no python3.12 in apt and conda's default channel was blocked (403) on the 4090 server, hence uv."
  run uv python install 3.12
  run uv venv --seed --python 3.12 "$VENV"
fi
export PIP_NO_CACHE_DIR=1 PIP_RETRIES=10 PIP_TIMEOUT=60 PIP_RESUME_RETRIES=10 PIP_DISABLE_PIP_VERSION_CHECK=1
if ! ((DRY)) && ! "$VENV_PY" -m pip install --help 2>/dev/null | grep -q -- '--resume-retries'; then
  say "pip in $VENV cannot resume interrupted downloads: upgrading pip inside the venv"
  run "$VENV_PY" -m pip install -U pip
fi

# --------------------------------------------------------------------------------------------------------------------
step 5 "torch ${TORCH_VER} / torchvision ${TV_VER}, CUDA build ${IDX}"
if ! ((DRY)) && gv torch-status --torch-version "$TORCH_VER" --torchvision-version "$TV_VER" --index "$IDX" 2>/dev/null; then
  say "already installed: torch ${TORCH_VER}+${IDX}"
else
  mkdir -p "$WHEELHOUSE"
  say "wheels are downloaded into $WHEELHOUSE first: a finished wheel is never fetched twice, an interrupted one is resumed (pip --resume-retries)."
  attempt=1
  # PIP_CONFIG_FILE=/dev/null: the venv's pip.conf (written below) pins the torch that is installed NOW, which must not block a change.
  until run env PIP_CONFIG_FILE=/dev/null "$VENV_PY" -m pip download "torch==${TORCH_VER}+${IDX}" "torchvision==${TV_VER}+${IDX}" \
        --index-url "https://download.pytorch.org/whl/${IDX}" --extra-index-url https://pypi.org/simple -d "$WHEELHOUSE"; do
    ((attempt < 5)) || die "the torch download failed 5 times. Finished wheels stay in $WHEELHOUSE; run the script again to continue."
    attempt=$((attempt + 1)); say "download interrupted; retry $attempt of 5 in 10 s"; sleep 10
  done
  run env PIP_CONFIG_FILE=/dev/null "$VENV_PY" -m pip install --no-index --find-links "$WHEELHOUSE" "torch==${TORCH_VER}+${IDX}" "torchvision==${TV_VER}+${IDX}"
fi
run "$VENV_PY" "$HELPER" write-constraints --out "$VENV/torch-constraints.txt" --pip-conf "$VENV/pip.conf"
CONSTRAINT=(-c "$VENV/torch-constraints.txt")

# --------------------------------------------------------------------------------------------------------------------
step 6 "Project requirements (torch stays pinned by the constraints file)"
run "$VENV_PY" -m pip install "${CONSTRAINT[@]}" -r "$REPO/requirements.txt" python-dotenv
run "$VENV_PY" -m pip install "${CONSTRAINT[@]}" -r "$REPO/requirements-gpu.txt"
rec packages "requirements.txt + requirements-gpu.txt + python-dotenv; torch/torchvision pinned in $VENV/torch-constraints.txt (also in $VENV/pip.conf, so '!pip install' in a notebook cannot replace them)"

# --------------------------------------------------------------------------------------------------------------------
step 7 ".env and the Hugging Face token"
run "$VENV_PY" "$HELPER" ensure-env --repo "$REPO" --state "$STATE"
run "$VENV_PY" "$HELPER" check-hf --repo "$REPO" --state "$STATE"

# --------------------------------------------------------------------------------------------------------------------
step 8 "Install the project and check torch / bitsandbytes"
run "$VENV_PY" -m pip install -e "$REPO" --no-deps
if ((DRY)); then
  say "  (dry-run) pip check; python -m bitsandbytes; verify torch; verify bnb"
else
  if out="$("$VENV_PY" -m pip check 2>&1)"; then
    rec project "dental-agent installed in editable mode (--no-deps); pip check: no broken requirements"
  else
    printf '%s\n' "$out"
    rec project "dental-agent installed in editable mode; pip check reported problems (listed above in the log)"
  fi
  mkdir -p "$STATE_DIR/logs"
  if "$VENV_PY" -m bitsandbytes >"$STATE_DIR/logs/bitsandbytes.log" 2>&1; then say "python -m bitsandbytes: ok (log: $STATE_DIR/logs/bitsandbytes.log)"
  else warn "python -m bitsandbytes reported a problem; last lines:"; tail -n 12 "$STATE_DIR/logs/bitsandbytes.log" >&2 || true; fi
  gv verify torch --state "$STATE" || { HARD_FAIL=1; miss "torch cannot run on this GPU (see 'torch check')"; }
  gv verify bnb --state "$STATE" >/dev/null || miss "bitsandbytes 4-bit forward failed: QLoRA (4-bit) is not available on this machine"
fi

# --------------------------------------------------------------------------------------------------------------------
step 9 "Attention kernels and accelerators (from compute capability)"
FA_BY_US=0
if [[ "$PLAN_FA" == "true" || -n "$FA_WHEEL" ]]; then
  if ((DRY)); then
    say "  (dry-run) flash-attn: reuse an importable one, else $( [[ -n "$FA_WHEEL" ]] && echo 'the wheel you passed' || echo 'an official prebuilt wheel matching torch/CUDA/Python/ABI'); then verify"
  elif "$VENV_PY" -c 'import flash_attn' 2>/dev/null; then
    say "flash-attn is already importable (not installed by this run); it is verified below."
  else
    fa_url="$FA_WHEEL"
    if [[ -z "$fa_url" ]]; then fa_url="$(gv find-flash-attn-wheel --state "$STATE" 2>/dev/null || true)"; fi
    if [[ -n "$fa_url" ]]; then
      "$VENV_PY" -m pip install "${CONSTRAINT[@]}" --no-deps "$fa_url" && FA_BY_US=1 || warn "the flash-attn wheel did not install"
    else
      say "no official prebuilt flash-attn wheel for this torch/CUDA/Python/ABI: $(gs get --state "$STATE" flash_attn_search 2>/dev/null)."
      say "Building it from source needs nvcc and takes long; this script does not do that. PyTorch SDPA is used."
      hint="$(gs commented flash-wheel --requirements-gpu "$REPO/requirements-gpu.txt" 2>/dev/null || true)"
      [[ -n "$hint" ]] && say "requirements-gpu.txt keeps a commented THIRD-PARTY wheel; to try it (your call):  bash scripts/setup_gpu_server.sh --flash-attn-wheel '$hint'"
    fi
  fi
else
  say "flash-attn: not applicable or switched off (see the plan in step 2); PyTorch SDPA is used."
fi
if ! ((DRY)); then
  gv verify attention --state "$STATE" >/dev/null || true
  fa_check="$(gs get --state "$STATE" flash_attn_check)"
  rec flash_attn "$fa_check$( ((FA_BY_US)) && echo ' (installed by this run)')"
  if ((FA_BY_US)) && [[ "$fa_check" == fail* || "$fa_check" == "import failed"* ]]; then
    warn "flash-attn installed but did not pass a real forward/backward pass; removing the package this run installed"
    "$VENV_PY" -m pip uninstall -y flash-attn || true
    rec flash_attn "removed again: $fa_check"
    miss "flash-attn: $fa_check"
  fi
  fa_imports=false; "$VENV_PY" -c 'import flash_attn' 2>/dev/null && fa_imports=true
  gs attn-outcome --state "$STATE" --flash-imports "$fa_imports"
  say "PyTorch SDPA backends that ran a forward+backward pass here: $(gs get --state "$STATE" sdpa)"
fi

# flash-linear-attention: installed from requirements-gpu.txt in step 6; the check mirrors how transformers imports it.
if ! ((DRY)); then
  gv verify fla --state "$STATE" >/dev/null || miss "flash-linear-attention: $(gs get --state "$STATE" fla) -> Qwen3.5 would run its slow reference PyTorch path"
  say "flash-linear-attention: $(gs get --state "$STATE" fla)"
fi

# causal-conv1d: prebuilt wheel first, else build only when an nvcc of the same CUDA major as torch exists.
if ((NO_CC1D)); then
  rec causal_conv1d "skipped (--no-causal-conv1d)"
elif ((DRY)); then
  say "  (dry-run) causal-conv1d: official wheel if one exists for this torch, else build when nvcc matches, else report"
else
  CC_BY_US=0
  if gv verify causal-conv1d --state "$STATE" >/dev/null 2>&1; then
    say "causal-conv1d: $(gs get --state "$STATE" causal_conv1d)"
  else
    CC_VER="$(gs commented causal-conv1d --requirements-gpu "$REPO/requirements-gpu.txt" 2>/dev/null || true)"
    if [[ -z "$CC_VER" ]]; then
      rec causal_conv1d "not installed: requirements-gpu.txt names no causal-conv1d version"
    else
      cc_url="$(gv find-causal-conv1d-wheel --version "$CC_VER" --state "$STATE" 2>/dev/null || true)"
      if [[ -n "$cc_url" ]]; then
        "$VENV_PY" -m pip install "${CONSTRAINT[@]}" --no-deps "$cc_url" && CC_BY_US=1 || warn "the causal-conv1d wheel did not install"
      else
        torch_cuda_major="$(gv env-tags | "$VENV_PY" -c 'import json,sys; print(json.load(sys.stdin)["cuda_major"])')"
        NVCC="$(command -v nvcc 2>/dev/null || true)"
        [[ -z "$NVCC" && -x /usr/local/cuda/bin/nvcc ]] && NVCC=/usr/local/cuda/bin/nvcc
        nvcc_major=""
        [[ -n "$NVCC" ]] && nvcc_major="$("$NVCC" --version 2>/dev/null | sed -n 's/.*release \([0-9][0-9]*\)\..*/\1/p' | head -n1)"
        if [[ -n "$nvcc_major" && "$nvcc_major" == "$torch_cuda_major" ]]; then
          cores="$(nproc)"; ram_gb="$(awk '/MemTotal/ {print int($2/1048576)}' /proc/meminfo)"
          jobs=$((ram_gb / 4)); ((jobs > cores / 2)) && jobs=$((cores / 2)); ((jobs < 1)) && jobs=1
          say "no official wheel ($(gs get --state "$STATE" causal_conv1d_search)); building with $NVCC (CUDA $nvcc_major), MAX_JOBS=$jobs"
          "$VENV_PY" -m pip install "${CONSTRAINT[@]}" setuptools wheel packaging ninja
          env CAUSAL_CONV1D_FORCE_BUILD=TRUE MAX_JOBS="$jobs" CUDA_HOME="$(dirname "$(dirname "$NVCC")")" \
            "$VENV_PY" -m pip install "${CONSTRAINT[@]}" "causal-conv1d==$CC_VER" --no-build-isolation --no-deps \
            && CC_BY_US=1 || warn "building causal-conv1d failed"
        else
          rec causal_conv1d "MISSING: no official wheel ($(gs get --state "$STATE" causal_conv1d_search)) and no nvcc of CUDA major ${torch_cuda_major} (found: ${NVCC:-none}${nvcc_major:+, CUDA $nvcc_major})"
          miss "causal-conv1d kernel (optional; Qwen3.5 then uses PyTorch's conv1d for that op). To build it install a CUDA ${torch_cuda_major} toolkit that provides nvcc (a system change this script does not make) and run the script again"
        fi
      fi
      if "$VENV_PY" -c 'import causal_conv1d' 2>/dev/null; then
        if ! gv verify causal-conv1d --state "$STATE" >/dev/null 2>&1; then
          if ((CC_BY_US)); then
            warn "causal-conv1d installed but failed its check; removing the package this run installed"
            "$VENV_PY" -m pip uninstall -y causal-conv1d || true
            miss "causal-conv1d: $(gs get --state "$STATE" causal_conv1d)"
          fi
        fi
      fi
    fi
    say "causal-conv1d: $(gs get --state "$STATE" causal_conv1d)"
  fi
fi

# --------------------------------------------------------------------------------------------------------------------
step 10 "Jupyter kernel for the notebooks"
if ((NO_KERNEL)); then
  rec kernelspec "skipped (--no-kernel)"
else
  run "$VENV_PY" -m pip install "${CONSTRAINT[@]}" ipykernel
  # PATH is put first inside the kernel so that '!python' and '!pip' in notebook cells use this venv, not the system Python.
  run "$VENV_PY" -m ipykernel install --user --name vlm-dental --display-name "VLM-DENTAL (.venv)" \
      --env PATH "$VENV/bin:\${PATH}" --env VIRTUAL_ENV "$VENV"
  rec kernelspec "vlm-dental (\"VLM-DENTAL (.venv)\"), PATH starts with $VENV/bin"
fi

# --------------------------------------------------------------------------------------------------------------------
step 11 "Base model, traces and trace images (existing code in dental_agent/utils/inputs.py)"
MODEL_PATH=""
if ((SKIP_MODEL && SKIP_DATA)); then
  say "skipped (--skip-model --skip-data)"
elif ((DRY)); then
  say "  (dry-run) resolve_qwen_model(), resolve_stage_traces(), ensure_trace_images() for stage $STAGE / track $TRACK"
else
  fetch_args=(--repo "$REPO" --stage "$STAGE" --track "$TRACK" --state "$STATE")
  ((SKIP_MODEL)) && fetch_args+=(--no-model)
  ((SKIP_DATA)) && fetch_args+=(--no-data)
  if gv fetch-data "${fetch_args[@]}"; then
    :
  else
    HARD_FAIL=1
    miss "model / traces / images: the download did not finish (message above). Run the script again: it resumes."
  fi
  MODEL_PATH="$(gs get --state "$STATE" model_path)"
fi

# --------------------------------------------------------------------------------------------------------------------
step 12 "SFT preflight (read-only: loads no model)"
if ((SKIP_PREFLIGHT)) || ((DRY)); then
  say "skipped"
else
  pf_args=(--repo "$REPO" --stage "$STAGE" --track "$TRACK")
  [[ -n "$MODEL_PATH" ]] && pf_args+=(--model-id "$MODEL_PATH")
  mapfile -t PF < <(gv preflight-argv "${pf_args[@]}" -- ${PREFLIGHT_EXTRA[@]+"${PREFLIGHT_EXTRA[@]}"})
  say "(discovery flags only, as the notebook's GPU path passes them; precision and every training setting stay at train_sft.py's defaults."
  say " The notebook's own preflight, run by its launch cell with your real settings, is the authoritative one.)"
  say "+ python ${PF[*]}"
  mkdir -p "$STATE_DIR/logs"
  expect_status
  (cd "$REPO" && "$VENV_PY" "${PF[@]}") 2>&1 | tee "$STATE_DIR/logs/sft-preflight.log"
  pf_rc=${PIPESTATUS[0]}
  end_expect
  pf_line="$(grep -E 'Preflight result' "$STATE_DIR/logs/sft-preflight.log" | tail -n1 || true)"
  rec preflight "${pf_line:-no result line} (exit $pf_rc; log: $STATE_DIR/logs/sft-preflight.log)"
  ((pf_rc == 0)) || { HARD_FAIL=1; miss "SFT preflight reported FAIL lines (above): fix them, then run the script again"; }
fi

# --------------------------------------------------------------------------------------------------------------------
step 13 "Summary"
if ((DRY)); then say "--dry-run: nothing was changed."; fi
if [[ -d "$WHEELHOUSE" ]]; then
  say "The torch wheelhouse ($(du -sh "$WHEELHOUSE" 2>/dev/null | cut -f1)) is kept at $WHEELHOUSE so a reinstall needs no download; delete it yourself when you no longer need it."
fi
rec next_step "Open notebooks/VLM_Dental_Colab_SFT.ipynb, choose the kernel 'VLM-DENTAL (.venv)', and run the code cells in this order (0-based indices in the .ipynb): 2 (repo root), 6 (hardware), 8 (secrets, Hub login, trace sync), 10 (configuration: set PRECISION yourself), 12 (token-length manifest), 20 (launch; it re-runs the preflight with your real settings). Cell 4 is the notebook's own install cell: it runs 'pip install -U' on several packages, so skipping it keeps the versions pinned in requirements.txt; torch stays pinned either way."
gs summary --state "$STATE"
exit "$HARD_FAIL"
