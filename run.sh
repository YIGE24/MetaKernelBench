#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"

usage() {
    cat <<'EOF'
Usage:
  ./run.sh <scope> <model> <mode> [cli options]

Scope:  all | attention | kv_cache | linear_attention | mlp_norm | moe | quantization
Model:  deepseek-pro | glm | glm-flash | gemini-flash | gpt | qwen-flash | any LiteLLM id
        (deepseek-pro and qwen-flash go to first-party APIs, the other aliases route through OpenRouter)
Mode:   tirx-solo | cutedsl-solo | cutedsl2tirx | tirx2cutedsl | all

Everything after the three positionals goes to metakernelbench.cli untouched.
Full paper run, one model:
  ./run.sh all glm all --n-replicates 1
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi
if [[ $# -lt 3 ]]; then
    usage >&2
    exit 2
fi
SCOPE="$1"
MODEL="$2"
MODE="$3"
shift 3

case "$MODEL" in
    deepseek-pro) MODEL="deepseek/deepseek-v4-pro" ;;
    glm) MODEL="openrouter/z-ai/glm-5.3" ;;
    glm-flash) MODEL="openrouter/z-ai/glm-5.3-flash" ;;
    gemini-flash) MODEL="openrouter/google/gemini-3.7-flash" ;;
    gpt) MODEL="openrouter/openai/gpt-5.6-sol" ;;
    qwen-flash) MODEL="dashscope/qwen3.8-flash" ;;
esac

case "$MODE" in
    tirx-solo) TRIALS="*_solo_tirx" ;;
    cutedsl-solo) TRIALS="*_solo_cutedsl" ;;
    cutedsl2tirx) TRIALS="*_cutedsl2tirx_skill" ;;
    tirx2cutedsl) TRIALS="*_tirx2cutedsl_skill" ;;
    all) TRIALS="*" ;;
    *) echo "unknown mode: $MODE" >&2; exit 2 ;;
esac

ARGS=(--model "$MODEL" --trials "$TRIALS")
if [[ "$SCOPE" != all ]]; then
    ARGS+=(--categories "$SCOPE")
fi

if [[ -f "$HOME/.config/mkbench/secrets.env" ]]; then
    set -a
    source "$HOME/.config/mkbench/secrets.env"
    set +a
fi

echo "model=$MODEL trials=$TRIALS scope=$SCOPE"
"$PYTHON_BIN" -m metakernelbench.build
"$PYTHON_BIN" -m metakernelbench.cli "${ARGS[@]}" "$@"
