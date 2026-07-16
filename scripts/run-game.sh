#!/usr/bin/env bash
# Starts the game backend and frontend together. Ctrl+C stops both.
#
# Shutdown is deliberately gentle. The backend has to reach its cleanup hook to
# release the RealSense camera and port 8000, and killing it mid-teardown is
# what makes librealsense dump core. So we ask everything to stop, give it room
# to do so, and only force-kill what is genuinely wedged.
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

self=$$
parent=$PPID
pgid=$(ps -o pgid= -p $self | tr -d ' ')

# Everything we spawn stays in our process group, even after a parent dies and
# the child reparents to init -- npm does not forward signals to the `sh -c`
# that actually runs vite, so tracking child PIDs alone leaks the node process.
# Signalling the group is what a terminal does for Ctrl+C. Never signal $self:
# a trap that signals its own group re-enters itself until bash's stack blows.
survivors() {
    local pid
    for pid in $(ps -o pid= -g "$pgid" 2>/dev/null); do
        [[ $pid == "$self" || $pid == "$parent" ]] && continue
        echo "$pid"
    done
}

signal_group() {
    local pid
    for pid in $(survivors); do
        kill "-$1" "$pid" 2>/dev/null
    done
}

shutdown() {
    trap - EXIT INT TERM

    signal_group TERM

    local deadline=$((SECONDS + 10))
    while [[ -n $(survivors) ]] && ((SECONDS < deadline)); do
        sleep 0.1
    done

    local left
    left=$(survivors)
    if [[ -n $left ]]; then
        echo "run-game: forcing $(echo "$left" | tr '\n' ' ')"
        signal_group KILL
    fi
}
trap shutdown EXIT INT TERM

# exec so the tracked PID is flask/npm itself rather than a wrapper subshell
# that can die while leaving the real server holding the port.
(cd robot-game/backend && exec flask --app main run --host 0.0.0.0 --port 8000) &
(cd robot-game/frontend && exec npm run dev -- --host) &

wait
