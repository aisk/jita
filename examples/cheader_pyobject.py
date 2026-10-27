"""Read CPython objects from generated code, with layouts from Python.h.

`jita.cheader.load` reads the interpreter's own headers and turns their
structs into ctypes types, so `typed(rsi, h.PyVarObject).ob_size` is the
same field the C API macro `Py_SIZE` reads, at the offset of this build
(the free threading build has a larger object header). The functions are
builtins with the `METH_O` calling convention: CPython calls `f(self,
arg)` with the GIL held, and the code boxes its result with the C API.

Needs `pip install "jita[cheader]"`. Run with
`uv run python examples/cheader_pyobject.py`.
"""

import ctypes
import sysconfig

from jita import METH_O, Extern, builtin
from jita.cheader import load, python_defines
from jita.x64 import *  # noqa: F403

# python_defines() adds the build flags pyconfig.h may not record, such as
# Py_GIL_DISABLED on Windows.
h = load("Python.h", include_dirs=[sysconfig.get_path("include")], defines=python_defines())


@builtin(METH_O)
def py_size() -> None:
    mov(rdi, typed(rsi, h.PyVarObject).ob_size)
    jmp(qword[rip + Extern(ctypes.pythonapi.PyLong_FromSsize_t)])


@builtin(METH_O)
def float_value() -> None:
    movsd(xmm0, typed(rsi, h.PyFloatObject).ob_fval)
    jmp(qword[rip + Extern(ctypes.pythonapi.PyFloat_FromDouble)])


print("ob_size offset:", h.PyVarObject.ob_size.offset)
print("list size:", py_size([1, 2, 3]))
print("tuple size:", py_size(("a", "b")))
print("float value:", float_value(2.75))
print("heap type flag:", h.Py_TPFLAGS_HEAPTYPE)
