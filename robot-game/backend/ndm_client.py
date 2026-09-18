"""
Client for the ndm-nfc dashboard's station API (see ndm-nfc/API.md).

Two things cross the network: a tag UID is looked up to get the numeric player
id when a game starts, and one result is submitted when it ends. Everything is
optional -- without NDM_BASE_URL and NDM_API_KEY the module stays inert and the
game runs exactly as before, which is what a demo without the dashboard needs.

Results go through a queue that is persisted to disk. A game that has been
played is a fact the dashboard has to hear about eventually, so a WLAN hiccup
at the moment the fourth token lands must not drop it, and neither must a
restart of this backend. The submission UUID is minted once per attempt and
reused on every retry, which is what makes the dashboard's idempotency work:
the same UUID with the same payload is answered with the original result
instead of creating a second one.

urllib is used rather than requests so this needs no dependency beyond the
standard library -- the pixi environment does not carry requests.
"""

import json
import os
import threading
import time
import urllib.error
import urllib.request
import uuid

BASE_URL = os.environ.get("NDM_BASE_URL", "").strip().rstrip("/")
API_KEY = os.environ.get("NDM_API_KEY", "").strip()
GAME_ID = os.environ.get("NDM_GAME_ID", "vier_gewinnt").strip()

QUEUE_FILE = os.environ.get(
    "NDM_QUEUE_FILE",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "ndm_queue.json"),
)

# The tag lookup sits between the player putting the card down and the UI
# saying "registriert", so it gets a short leash: better to report an unknown
# player quickly than to make someone wait on a server that is not answering.
LOOKUP_TIMEOUT = 3.0
# The submission happens after the game, out of anyone's sight, and is retried
# anyway, so it can afford to be patient.
SUBMIT_TIMEOUT = 8.0

# Local difficulty names (robot_controller.DIFFICULTY_NAMES) translated to the
# difficulty ids the dashboard is configured with -- "unmöglich" is spelled
# with the umlaut there. NDM_DIFFICULTY_MAP overrides any entry as JSON should
# the server's ids change:
#   NDM_DIFFICULTY_MAP='{"impossible":"sehr_schwer"}'
# An id the server does not know is rejected with 422, and those submissions
# wait in the queue rather than being thrown away -- see _attempt().
DEFAULT_DIFFICULTY_MAP = {
    "easy": "leicht",
    "medium": "mittel",
    "hard": "schwer",
    "impossible": "unmöglich",
}

# check_winner() reports 1 (human), 2 (robot) or 0 (draw). The API documents
# only win and loss; draw is being added on the server side, so it is sent
# under this id and overridable via NDM_OUTCOME_DRAW should it end up named
# differently.
OUTCOME_DRAW = os.environ.get("NDM_OUTCOME_DRAW", "draw").strip() or "draw"

# Retry pacing. Transport failures back off geometrically up to RETRY_MAX.
RETRY_BASE = 2.0
RETRY_MAX = 60.0
# A rejection that only a human can fix -- a wrong API key, or a difficulty and
# outcome the server does not know yet -- is not worth hammering. It is still
# retried, because the fix (correcting the key, adding the difficulty) happens
# on the server while this process keeps running, and the queued results should
# land by themselves once it does.
RETRY_NEEDS_ADMIN = 120.0
# Beyond this the oldest entries are dropped. Reaching it means the dashboard
# has been unreachable for hundreds of games, at which point the disk file is
# the problem and not the data.
MAX_QUEUE = 500

_lock = threading.Lock()
_queue = []
_worker_started = False
_wake = threading.Event()

_status = {
    "enabled": False,
    "base_url": None,
    "game": GAME_ID,
    "queue_length": 0,
    "last_error": None,
    "last_submit_at": None,
    "server_difficulties": None,
}


def is_enabled():
    """True when the dashboard is configured; everything else no-ops without it."""
    return bool(BASE_URL and API_KEY)


def difficulty_map():
    mapping = dict(DEFAULT_DIFFICULTY_MAP)
    raw = os.environ.get("NDM_DIFFICULTY_MAP", "").strip()
    if raw:
        try:
            override = json.loads(raw)
            if isinstance(override, dict):
                mapping.update({str(k): str(v) for k, v in override.items()})
            else:
                print("[NDM] NDM_DIFFICULTY_MAP is not a JSON object, ignoring it")
        except ValueError as e:
            print(f"[NDM] NDM_DIFFICULTY_MAP is not valid JSON ({e}), ignoring it")
    return mapping


def server_difficulty(local_name):
    """
    The dashboard's difficulty id for a local difficulty name, or None if the
    name has no mapping at all (which is a bug, not a server state).
    """
    return difficulty_map().get(local_name)


def outcome_for_winner(winner):
    """
    Translates check_winner()'s value into the API's outcome, from the player's
    perspective: the human is player 1, the robot is player 2.
    """
    if winner == 1:
        return "win"
    if winner == 2:
        return "loss"
    if winner == 0:
        return OUTCOME_DRAW
    return None


def _update_status(**changes):
    with _lock:
        _status.update(changes)


def _request(method, path, payload=None, timeout=SUBMIT_TIMEOUT):
    """
    One HTTP call against the station API. Returns (status, parsed_body).
    Raises urllib.error.URLError and friends for transport failures.
    """
    url = f"{BASE_URL}{path}"
    data = None
    headers = {"X-API-Key": API_KEY, "Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            status = resp.status
    except urllib.error.HTTPError as e:
        # An HTTP error is an answer, not a failure to reach the server, so the
        # status code is what the caller has to decide on.
        body = e.read().decode("utf-8", errors="replace")
        status = e.code

    try:
        parsed = json.loads(body) if body else None
    except ValueError:
        parsed = None
    return status, parsed


def lookup_tag(uid):
    """
    Resolves a tag UID to {"id": int, "name": str}.

    Returns None when the tag is unknown to the dashboard, and raises on a
    transport failure so the caller can tell "not registered" apart from
    "server unreachable" -- the player needs different advice in each case.
    """
    if not is_enabled():
        return None

    status, body = _request("GET", f"/api/tags/{uid}", timeout=LOOKUP_TIMEOUT)
    if status == 200 and isinstance(body, dict) and isinstance(body.get("id"), int):
        _update_status(last_error=None)
        return {"id": body["id"], "name": body.get("name")}
    if status == 404:
        _update_status(last_error=None)
        return None
    message = f"Tag lookup failed with HTTP {status}"
    _update_status(last_error=message)
    raise RuntimeError(message)


def fetch_games():
    """
    Reads /api/games and records the difficulties the server knows for our game
    id. Used only to warn about mappings the dashboard would reject; never
    blocks a game from being played.
    """
    if not is_enabled():
        return None
    try:
        status, body = _request("GET", "/api/games", timeout=LOOKUP_TIMEOUT)
    except Exception as e:
        print(f"[NDM] Could not read /api/games: {e}")
        return None
    if status != 200 or not isinstance(body, dict):
        print(f"[NDM] /api/games answered HTTP {status}")
        return None

    for game in body.get("games", []):
        if isinstance(game, dict) and game.get("id") == GAME_ID:
            difficulties = [str(d) for d in game.get("difficulties", [])]
            _update_status(server_difficulties=difficulties)
            missing = sorted(
                {v for v in difficulty_map().values()} - set(difficulties)
            )
            if missing:
                print(
                    f"[NDM] Dashboard does not (yet) know difficulty ids "
                    f"{missing} for game '{GAME_ID}'; it offers {difficulties}. "
                    f"Results on those difficulties stay queued until it does."
                )
            return difficulties

    print(
        f"[NDM] Dashboard does not know game id '{GAME_ID}'. "
        f"Set NDM_GAME_ID to one of "
        f"{[g.get('id') for g in body.get('games', []) if isinstance(g, dict)]}."
    )
    return None


def _load_queue():
    try:
        with open(QUEUE_FILE) as f:
            stored = json.load(f)
    except FileNotFoundError:
        return []
    except (OSError, ValueError) as e:
        print(f"[NDM] Could not read {QUEUE_FILE} ({e}); starting with an empty queue")
        return []

    if not isinstance(stored, list):
        print(f"[NDM] Ignoring malformed {QUEUE_FILE}")
        return []

    entries = []
    for item in stored:
        if not isinstance(item, dict) or not isinstance(item.get("payload"), dict):
            continue
        # A restart is a good moment to try again, whatever backoff the entry
        # was serving when the process went down.
        item["next_attempt_at"] = 0.0
        entries.append(item)
    return entries


def _save_queue_locked():
    tmp_path = f"{QUEUE_FILE}.tmp"
    try:
        with open(tmp_path, "w") as f:
            json.dump(_queue, f, indent=2)
        os.replace(tmp_path, QUEUE_FILE)
    except OSError as e:
        print(f"[NDM] Could not persist the result queue: {e}")


def submit_result(player_id, difficulty, duration_ms, outcome):
    """
    Queues one finished game for the dashboard and returns its submission id.

    The call does not block on the network: the payload is written to the queue
    file and a background worker delivers it. Returns None when the dashboard
    is not configured or the arguments do not describe a submittable game.
    """
    if not is_enabled():
        return None
    if not isinstance(player_id, int) or player_id <= 0:
        return None
    if not difficulty or not outcome:
        print(
            f"[NDM] Not submitting: difficulty={difficulty!r} outcome={outcome!r} "
            f"is incomplete"
        )
        return None

    duration_ms = int(max(1, round(duration_ms)))
    submission_id = str(uuid.uuid4())
    entry = {
        "payload": {
            "submission_id": submission_id,
            "player_id": player_id,
            "game": GAME_ID,
            "difficulty": difficulty,
            "duration_ms": duration_ms,
            "outcome": outcome,
        },
        "attempts": 0,
        "next_attempt_at": 0.0,
        "last_error": None,
        "queued_at": time.time(),
    }

    with _lock:
        _queue.append(entry)
        if len(_queue) > MAX_QUEUE:
            dropped = len(_queue) - MAX_QUEUE
            del _queue[:dropped]
            print(f"[NDM] Result queue over {MAX_QUEUE}, dropped {dropped} oldest")
        _save_queue_locked()
        _status["queue_length"] = len(_queue)

    print(
        f"[NDM] Queued result: player={player_id} difficulty={difficulty} "
        f"outcome={outcome} duration={duration_ms}ms id={submission_id}"
    )
    _wake.set()
    return submission_id


def _attempt(entry):
    """
    Sends one queued entry. Returns True when it is done with (delivered, or
    rejected in a way a retry cannot mend) and can leave the queue.
    """
    payload = entry["payload"]
    entry["attempts"] += 1

    try:
        status, body = _request("POST", "/api/results", payload)
    except Exception as e:
        # Transport failure: the dashboard may never have seen it, so the exact
        # same payload and UUID go out again later.
        entry["last_error"] = f"{type(e).__name__}: {e}"
        entry["next_attempt_at"] = time.time() + min(
            RETRY_MAX, RETRY_BASE * (2 ** min(entry["attempts"] - 1, 5))
        )
        _update_status(last_error=entry["last_error"])
        return False

    if status == 200:
        created = body.get("created") if isinstance(body, dict) else None
        removed = body.get("removed") if isinstance(body, dict) else None
        note = "" if created else " (already recorded)"
        if removed:
            note = " (previously removed by an admin, not recreated)"
        print(f"[NDM] Result {payload['submission_id']} accepted{note}")
        _update_status(last_error=None, last_submit_at=time.time())
        return True

    if status == 409:
        # The UUID is minted per attempt and never reused with different data,
        # so this cannot happen from here. If it does, retrying is pointless and
        # the stored attempt must not be overwritten; drop it and say so.
        print(
            f"[NDM] Result {payload['submission_id']} rejected as a conflicting "
            f"duplicate (409). Dropping it; the dashboard keeps its original."
        )
        _update_status(last_error="409 conflicting duplicate")
        return True

    if status in (401, 422):
        # 401 is a wrong key, 422 a value the dashboard does not know -- the
        # draw outcome and the fourth difficulty until they are added there.
        # Both are fixed on the other side while this keeps running, so the
        # entry waits instead of being discarded.
        entry["last_error"] = f"HTTP {status}: {body}"
        entry["next_attempt_at"] = time.time() + RETRY_NEEDS_ADMIN
        reason = (
            "API key rejected"
            if status == 401
            else f"server rejected difficulty/outcome "
            f"({payload['difficulty']}/{payload['outcome']})"
        )
        if entry["attempts"] == 1 or entry["attempts"] % 10 == 0:
            print(
                f"[NDM] {reason} for {payload['submission_id']} (HTTP {status}); "
                f"keeping it queued, attempt {entry['attempts']}"
            )
        _update_status(last_error=entry["last_error"])
        return False

    entry["last_error"] = f"HTTP {status}: {body}"
    entry["next_attempt_at"] = time.time() + min(
        RETRY_MAX, RETRY_BASE * (2 ** min(entry["attempts"] - 1, 5))
    )
    _update_status(last_error=entry["last_error"])
    return False


def _worker():
    fetch_games()
    while True:
        now = time.time()
        with _lock:
            due = [e for e in _queue if e["next_attempt_at"] <= now]

        # Each due entry is tried on its own rather than stopping at the first
        # failure: one result the server will not take yet (a draw before it
        # knows about draws) must not hold up every result behind it.
        for entry in due:
            done = _attempt(entry)
            with _lock:
                if done and entry in _queue:
                    _queue.remove(entry)
                _save_queue_locked()
                _status["queue_length"] = len(_queue)

        with _lock:
            pending = [e["next_attempt_at"] for e in _queue]
        if pending:
            delay = max(1.0, min(pending) - time.time())
        else:
            delay = 60.0
        _wake.wait(timeout=delay)
        _wake.clear()


def start():
    """
    Loads the queue from disk and starts the delivery worker. Safe to call when
    the dashboard is not configured: it then only reports that it is disabled.
    """
    global _worker_started

    _update_status(enabled=is_enabled(), base_url=BASE_URL or None, game=GAME_ID)
    if not is_enabled():
        print(
            "[NDM] Dashboard not configured (NDM_BASE_URL / NDM_API_KEY unset); "
            "tags are shown but no results are submitted."
        )
        return False

    with _lock:
        if _worker_started:
            return True
        _queue.extend(_load_queue())
        _status["queue_length"] = len(_queue)
        _worker_started = True
        pending = len(_queue)

    print(f"[NDM] Dashboard at {BASE_URL}, game '{GAME_ID}', {pending} result(s) queued")
    threading.Thread(target=_worker, daemon=True).start()
    return True


def status():
    """Snapshot for the status API."""
    with _lock:
        snapshot = dict(_status)
        snapshot["queue_length"] = len(_queue)
    return snapshot
