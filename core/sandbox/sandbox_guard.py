"""Best-effort guard that runs INSIDE the sandboxed Python process (standard library only).

It installs a Python audit hook that refuses network access, starting other programs, and
file access outside the sandbox folder. This is NOT a security boundary: audit hooks can be
bypassed by native code (for example ctypes) and by code that attacks the interpreter. The
guard stops accidental and naive access. For hostile code, use a container or a virtual
machine. Importing this module has no effect. Only install() activates the hook, and it must
be called only inside the disposable child process, never in the main application.
"""

from __future__ import annotations

import json
import os
import sys
import sysconfig
from dataclasses import dataclass

PROCESS_EVENTS = frozenset(
    {
        "subprocess.Popen", "os.system", "os.exec", "os.spawn", "os.posix_spawn",
        "os.fork", "os.forkpty", "os.startfile", "_winapi.CreateProcess",
    }
)
LINK_EVENTS = frozenset({"os.symlink", "os.link", "_winapi.CreateJunction"})
# event -> index of the argument that holds the address or host name
ADDRESS_EVENTS = {
    "socket.connect": 1, "socket.bind": 1, "socket.sendto": 1, "socket.sendmsg": 1,
    "socket.getaddrinfo": 0, "socket.gethostbyname": 0, "socket.gethostbyaddr": 0,
    "socket.getnameinfo": 0,
}
# event -> indexes of path arguments that are written to or removed
WRITE_PATHS = {
    "os.remove": (0,), "os.rmdir": (0,), "os.mkdir": (0,), "os.rename": (0, 1),
    "os.chmod": (0,), "os.chown": (0,), "os.utime": (0,), "shutil.copyfile": (1,),
    "shutil.copymode": (1,), "shutil.copystat": (1,), "shutil.copytree": (1,),
    "shutil.move": (1,), "shutil.rmtree": (0,),
}
# event -> indexes of path arguments that are read
READ_PATHS = {
    "os.listdir": (0,), "os.scandir": (0,), "shutil.copyfile": (0,),
    "shutil.copytree": (0,), "shutil.move": (0,),
}
LOOPBACK = frozenset({"localhost", "127.0.0.1", "::1"})
WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_TRUNC
DEVNULL_NAMES = frozenset({"nul", "/dev/null", "\\\\.\\nul"})


@dataclass(frozen=True)
class Policy:
    sandbox_root: str
    read_roots: tuple[str, ...]
    allow_loopback: bool = False


def normalize(path: object) -> str | None:
    """A comparable absolute path, or None when `path` is not a path (an int, None, ...)."""
    try:
        if isinstance(path, bytes):
            path = os.fsdecode(path)
        elif isinstance(path, str | os.PathLike):
            path = os.fspath(path)
            if isinstance(path, bytes):
                path = os.fsdecode(path)
        else:
            return None
        return os.path.normcase(os.path.realpath(path))
    except (TypeError, ValueError, OSError):
        return None


def inside(path: str, roots: tuple[str, ...]) -> bool:
    return any(path == root or path.startswith(root + os.sep) for root in roots)


def build_policy(sandbox_root: str, allow_loopback: bool = False) -> Policy:
    """Writes are allowed only inside `sandbox_root`. Reads also work for the Python install."""
    root = normalize(sandbox_root)
    if root is None:
        raise ValueError("sandbox_root is not a path")
    candidates = [*sys.path, sys.prefix, sys.base_prefix, sys.exec_prefix, sys.base_exec_prefix]
    candidates += list(sysconfig.get_paths().values())
    candidates += [os.environ.get(key, "") for key in ("SYSTEMROOT", "WINDIR")]
    roots = {root}
    for item in candidates:
        norm = normalize(item) if item else None
        if norm and os.path.dirname(norm) != norm:  # never allow a drive or filesystem root
            roots.add(norm)
    return Policy(root, tuple(sorted(roots)), allow_loopback)


def _is_loopback(host: object) -> bool:
    if isinstance(host, bytes):
        host = host.decode("ascii", "replace")
    if not isinstance(host, str):
        return False
    lowered = host.lower()
    return lowered in LOOPBACK or lowered.startswith("127.")


def _host(args: tuple, index: int) -> object:
    value = args[index] if len(args) > index else None
    if isinstance(value, tuple) and value:
        value = value[0]
    return value


def _writes(mode: object, flags: object) -> bool:
    if isinstance(flags, int) and flags & WRITE_FLAGS:
        return True
    return isinstance(mode, str) and any(char in mode for char in "wax+")


def _path_arg(args: tuple, index: int, cwd_if_none: bool = False) -> str | None:
    value = args[index] if len(args) > index else None
    if value is None and cwd_if_none:
        value = os.getcwd()
    return normalize(value)


def _open_reason(args: tuple, policy: Policy) -> str | None:
    path = args[0] if args else None
    if isinstance(path, int):
        return None
    if isinstance(path, str) and path.lower() in DEVNULL_NAMES:
        return None
    norm = normalize(path)
    if norm is None:
        return None
    mode = args[1] if len(args) > 1 else None
    flags = args[2] if len(args) > 2 else None
    if _writes(mode, flags):
        if not inside(norm, (policy.sandbox_root,)):
            return "writing outside the sandbox folder is blocked"
        return None
    if not inside(norm, policy.read_roots):
        return "reading files outside the sandbox folder is blocked"
    return None


def decide(event: str, args: tuple, policy: Policy) -> str | None:
    """Why the operation is refused, or None when it is allowed. Pure: safe to unit test."""
    if event in PROCESS_EVENTS:
        return "starting other programs is blocked"
    if event in LINK_EVENTS:
        return "creating links is blocked"
    if event in ADDRESS_EVENTS:
        host = _host(args, ADDRESS_EVENTS[event])
        if policy.allow_loopback and _is_loopback(host):
            return None
        return "network access is blocked"
    if event == "open":
        return _open_reason(args, policy)
    for index in WRITE_PATHS.get(event, ()):
        norm = _path_arg(args, index)
        if norm is not None and not inside(norm, (policy.sandbox_root,)):
            return "writing outside the sandbox folder is blocked"
    for index in READ_PATHS.get(event, ()):
        norm = _path_arg(args, index, cwd_if_none=True)
        if norm is not None and not inside(norm, policy.read_roots):
            return "reading files outside the sandbox folder is blocked"
    return None


def install(policy_path: str) -> None:
    """Activate the hook. Call only inside the disposable child process."""
    with open(policy_path, encoding="utf-8") as handle:
        data = json.load(handle)
    policy = build_policy(data["sandbox_root"], bool(data.get("allow_loopback", False)))

    def hook(event: str, args: tuple) -> None:
        try:
            reason = decide(event, args, policy)
        except Exception:  # noqa: BLE001 - a bug in the checker must not break unrelated work
            return
        if reason:
            raise PermissionError(f"sandbox: {reason} ({event})")

    sys.addaudithook(hook)