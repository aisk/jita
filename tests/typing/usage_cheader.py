"""Accepted jita.cheader usage: the names of a Header are Any, so they go
anywhere a ctypes type or an immediate does. Must type check cleanly
(tests/test_typing.py)."""

import ctypes
import sysconfig

from jita import function
from jita.cheader import Diagnostic, Header, HeaderError, diagnostics, load
from jita.x64 import *  # noqa: F403

h: Header = load(
    "Python.h",
    include_dirs=[sysconfig.get_path("include")],
    defines={"Py_LIMITED_API": 0x030E0000, "NDEBUG": None, "NAME": "x"},
    types={"off_t": ctypes.c_int64},
    names=["Py*", "_Py*"],
    exclude=["_*"],
    strict=False,
)


@function(ctypes.c_ssize_t, ctypes.c_void_p)
def py_len() -> None:
    mov(rax, typed(rdi, h.PyVarObject).ob_size)
    mov(ecx, h.Py_TPFLAGS_HEAPTYPE)
    ret()


obj_ptr = ctypes.POINTER(h.PyObject)
size: int = ctypes.sizeof(h.PyTypeObject) + h.PyVarObject.ob_size.offset
skipped: list[Diagnostic] = [d for d in diagnostics(h) if d.severity == "skip"]
where: str | None = skipped[0].file if skipped else None
try:
    load("missing.h")
except HeaderError as e:
    message = str(e)
