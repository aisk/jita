"""Read CPython objects from generated code, with layouts from Python.h.

`jita.cheader.load` reads the interpreter's own headers and turns their
structs into ctypes types, so `typed(rdi, h.PyVarObject).ob_size` is the
same field the C API macro `Py_SIZE` reads, at the offset of this build
(the free threading build has a larger object header). `id(obj)` is the
object's address in CPython.

Needs `pip install "jita[cheader]"`. Run with
`uv run python examples/cheader_pyobject.py`.
"""

import ctypes
import sysconfig

from jita import function
from jita.cheader import load, python_defines
from jita.x64 import *  # noqa: F403

# python_defines() adds the build flags pyconfig.h may not record, such as
# Py_GIL_DISABLED on Windows.
h = load("Python.h", include_dirs=[sysconfig.get_path("include")], defines=python_defines())


@function(ctypes.c_ssize_t, ctypes.c_void_p)
def py_size() -> None:
    mov(rax, typed(rdi, h.PyVarObject).ob_size)
    ret()


@function(ctypes.c_double, ctypes.c_void_p)
def float_value() -> None:
    movsd(xmm0, typed(rdi, h.PyFloatObject).ob_fval)
    ret()


print("ob_size offset:", h.PyVarObject.ob_size.offset)
print("list size:", py_size(id([1, 2, 3])))
print("tuple size:", py_size(id(("a", "b"))))
print("float value:", float_value(id(2.75)))
print("heap type flag:", h.Py_TPFLAGS_HEAPTYPE)
