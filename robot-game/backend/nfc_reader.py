import os
import re
import serial
import time
import threading

DEFAULT_PORT = os.environ.get("NFC_PORT", "/dev/ttyUSB0")
DEFAULT_BAUD = int(os.environ.get("NFC_BAUD", "115200"))

# The reader reports everything as plain CRLF terminated text, a scan as
# "Tag erkannt: 53:27:5C:9F:53:00:01" and problems in the same shape
# ("Auth fehlgeschlagen"), so only lines carrying the prefix are a tag.
# Lost bytes also glue messages together ("Auth fehlgeschlagenTagerkannt: ..."),
# so the prefix is searched anywhere in the line, tolerates a missing space and
# the last occurrence wins - that is the one followed by a complete uid.
TAG_PREFIX_PATTERN = re.compile(r"tag\s*(?:erkannt|detected)\s*:\s*", re.IGNORECASE)
# Bytes go missing on the wire now and then, which splices a truncated scan
# line onto the next message ("53:27:5C:" + "Auth fehlgeschlagen"). Only a uid
# of at least four hex bytes in one consistent notation is handed on, so
# damaged lines cannot reach the game as a fantasy tag id - a single dropped
# colon ("5327:5C:9F:53:00:01") is a damaged line, not a new player.
UID_PATTERN = re.compile(r"^[0-9A-Fa-f]{2}(?::[0-9A-Fa-f]{2}){3,}$"
                         r"|^[0-9A-Fa-f]{2}(?:-[0-9A-Fa-f]{2}){3,}$"
                         r"|^(?:[0-9A-Fa-f]{2}){4,}$")
# It repeats the line for as long as the tag rests on the antenna.
REPEAT_SUPPRESSION_SECONDS = 1.5
# Without a line end nothing is parseable, so a buffer this large is garbage
# (wrong baud rate) and gets dropped instead of growing forever.
MAX_BUFFER_BYTES = 4096

# Connection state of the serial reader. Written by nfc_worker, read by the
# status endpoints of main.py and calibration_server.py via the helpers below.
_state_lock = threading.Lock()
_reader_state = {
    "connected": False,
    "port": None,
    "baud": None,
    "error": None,
    "last_message": None,
}

def _update_state(**changes):
    with _state_lock:
        _reader_state.update(changes)

def parse_tag_line(line):
    """
    Classifies one reader line as ("tag", uid), ("damaged", line) for a scan
    line whose uid did not survive transmission, or ("message", line).
    """
    line = line.strip()
    prefixes = list(TAG_PREFIX_PATTERN.finditer(line))
    if not prefixes:
        return "message", line
    # Later prefixes first: in a spliced line the last one carries the uid that
    # arrived complete.
    for prefix in reversed(prefixes):
        uid = line[prefix.end():].strip()
        if UID_PATTERN.match(uid):
            return "tag", uid
    return "damaged", line

def nfc_worker(callback, port=DEFAULT_PORT, baud=DEFAULT_BAUD):
    """
    Background worker that continuously reads from the NFC serial port.
    """
    _update_state(port=port, baud=baud)
    while True:
        ser = None
        try:
            # Reconnect loop
            # Exclusive: main.py and calibration_server.py each start a reader,
            # and two of them on one tty silently split the byte stream into
            # garbage. Failing to open is the far more diagnosable outcome.
            ser = serial.Serial(port, baud, timeout=0.5, exclusive=True)
            print(f"[NFC] Connected to {port} at {baud} baud.")
            _update_state(connected=True, error=None)

            buffer = bytearray()
            last_tag = None
            last_tag_at = 0.0
            while True:
                try:
                    chunk = ser.read(max(ser.in_waiting, 1))
                    if chunk:
                        buffer.extend(chunk)
                    if len(buffer) > MAX_BUFFER_BYTES:
                        print(f"[NFC] No line ends in {len(buffer)} bytes, dropping "
                              f"(wrong baud rate?): {bytes(buffer[:32]).hex(' ')}...")
                        buffer.clear()

                    while b"\n" in buffer:
                        raw_line, _, rest = bytes(buffer).partition(b"\n")
                        buffer = bytearray(rest)
                        line = raw_line.decode("utf-8", errors="replace").strip()
                        if not line:
                            continue
                        _update_state(last_message=line)
                        kind, value = parse_tag_line(line)
                        if kind == "damaged":
                            print(f"[NFC] Ignoring damaged scan line: {line}")
                            continue
                        if kind != "tag":
                            print(f"[NFC] Reader message: {line}")
                            continue
                        tag = value
                        now = time.monotonic()
                        if tag == last_tag and now - last_tag_at < REPEAT_SUPPRESSION_SECONDS:
                            last_tag_at = now
                            continue
                        last_tag, last_tag_at = tag, now
                        print(f"[NFC] Tag scanned: {tag}")
                        callback(tag)
                except OSError as e:
                    # Device disconnected or error
                    print(f"[NFC] Device disconnected, reconnecting...")
                    _update_state(connected=False, error=str(e))
                    break

        except Exception as e:
            # Reader might be unplugged, wait before retrying
            _update_state(connected=False, error=str(e))
            time.sleep(2)
        finally:
            _update_state(connected=False)
            if ser is not None:
                try:
                    ser.close()
                except:
                    pass

def nfc_reader_connected():
    """
    True while the serial port to the reader is open.
    """
    with _state_lock:
        return _reader_state["connected"]

def reader_connection():
    """
    Snapshot of the reader connection (connected, port, baud, last error and
    last received line) for the status APIs.
    """
    with _state_lock:
        return dict(_reader_state)

def start_nfc_reader(callback):
    """
    Starts the NFC reader in a daemon thread.
    Returns the thread object.
    """
    t = threading.Thread(target=nfc_worker, args=(callback,), daemon=True)
    t.start()
    return t
