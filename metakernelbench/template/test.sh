set -eu

kernel=/app/kernel.py
submissions_dir=/app/.submissions
poll_secs=2
wait_secs=1200
heartbeat_secs=30

if [ ! -s "$kernel" ]; then
    echo "nothing to submit: $kernel is missing or empty"
    exit 1
fi
if ! python -c 'import ast, sys; ast.parse(open(sys.argv[1]).read(), sys.argv[1])' "$kernel"; then
    echo "not submitted: $kernel fails to parse, no submission was consumed"
    exit 1
fi

stamp=$(date +%s%N)
result="$submissions_dir/$stamp.result"
pending_note="the report will still arrive; read it later with cat $result, a re-run spends another submission"
mkdir -p "$submissions_dir"
cp "$kernel" "$submissions_dir/$stamp.tmp"
mv "$submissions_dir/$stamp.tmp" "$submissions_dir/$stamp.request"
echo "submitted $kernel, waiting for the GPU report"
trap 'echo "wait aborted: $pending_note"; exit 2' INT TERM

elapsed=0
while [ ! -f "$result" ]; do
    if [ "$elapsed" -ge "$wait_secs" ]; then
        echo "no result after ${wait_secs}s, the runner may be backed up"
        echo "$pending_note"
        exit 2
    fi
    if [ "$elapsed" -gt 0 ] && [ $((elapsed % heartbeat_secs)) -eq 0 ]; then
        echo "still waiting after ${elapsed}s"
    fi
    sleep "$poll_secs"
    elapsed=$((elapsed + poll_secs))
done
echo "report arrived after ${elapsed}s"
cat "$result"
