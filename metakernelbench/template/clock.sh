mkbench_clock() {
    read -r started budget < /app/.started 2>/dev/null || { printf 'clock not started'; return; }
    used=$(( $(date +%s) - started ))
    left=$(( budget - used ))
    printf '%dm used, %dm left' $(( used / 60 )) $(( left > 0 ? left / 60 : 0 ))
}
PS1='[$(mkbench_clock)] \u@\h:\w# '
