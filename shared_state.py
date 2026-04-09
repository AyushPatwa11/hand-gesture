from threading import Lock

_state = {
    "running": True,
    "bloom": True,
    "sound": False,
    "save_counter": 0,
    "clear_counter": 0,
    "capture_index": 0,
}
_lock = Lock()


def get(key):
    with _lock:
        return _state.get(key)


def set(key, value):
    with _lock:
        _state[key] = value


def toggle(key):
    with _lock:
        _state[key] = not _state.get(key, False)
        return _state[key]


def request_save():
    with _lock:
        _state["save_counter"] += 1
        return _state["save_counter"]


def request_clear():
    with _lock:
        _state["clear_counter"] += 1
        return _state["clear_counter"]


def set_running(val: bool):
    set("running", bool(val))


def is_running():
    return get("running")
