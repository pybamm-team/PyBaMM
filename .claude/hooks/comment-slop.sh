#!/bin/sh
# PostToolUse wrapper for comment-slop, reporting on the lines just written.
set -u

PIN="comment-slop==0.1.0"
# Read once and replayed, since uv consumes stdin before any fallback runs.
payload=$(cat)
out=""

run() {
    printf '%s' "$payload" | "$@" -m comment_slop --hook 2>&1
}

# Every interpreter exits 2 on its own failures too, so the protocol's "block
# and show the agent this" status is re-derived rather than passed through.
is_findings() {
    case "$out" in
        "comment-slop: "*) return 0 ;;
    esac
    return 1
}

status=1
# --quiet: uv's "Installed 1 package" would land ahead of the findings and
# defeat the prefix check above.
if command -v uv >/dev/null 2>&1; then
    out=$(run uv run --no-project --quiet --with "$PIN" --python ">=3.11" python)
    status=$?
fi
# A clean exit means a clean file; anything else is uv's own failure, so retry
# on a python3 that already has the package (`pip install comment-slop`).
if ! is_findings && [ "$status" -ne 0 ] && command -v python3 >/dev/null 2>&1
then
    out=$(run python3)
fi

[ -n "$out" ] && printf '%s\n' "$out" >&2
is_findings && exit 2
exit 0
