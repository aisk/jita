"""jita.cheader on the running interpreter's own Python.h: the layouts are
checked against live objects, so nothing depends on the build (the GIL
and free threading builds differ) or needs a C compiler."""

import ctypes
import platform
import sys
import sysconfig
from pathlib import Path

import pytest

pytest.importorskip("pycparser")
pytest.importorskip("pcpp")

from jita import function  # noqa: E402
from jita.cheader import Header, diagnostics, load, python_defines  # noqa: E402

INCLUDE = sysconfig.get_path("include")
pytestmark = pytest.mark.skipif(not (Path(INCLUDE) / "Python.h").is_file(), reason="Python.h is not installed")

# System types some prototypes and Py_tss_t use by value.
SYSTEM_TYPES = {"va_list": ctypes.c_void_p, "pthread_key_t": ctypes.c_uint}


@pytest.fixture(scope="module")
def h() -> Header:
    return load("Python.h", include_dirs=[INCLUDE])


def test_only_expected_skips(h):
    # A new construct in a future Python.h shows up here.
    for d in diagnostics(h):
        if d.severity == "skip":
            assert d.message == "variadic function" or "system type of unknown size" in d.message, d


def test_only_variadic_functions_skipped_with_system_types():
    h = load("Python.h", include_dirs=[INCLUDE], types=SYSTEM_TYPES)
    skipped = [d for d in diagnostics(h) if d.severity == "skip"]
    assert skipped and all(d.message == "variadic function" for d in skipped), skipped
    assert "PyErr_Format" in {d.name for d in skipped}
    with pytest.raises(AttributeError, match=r"PyErr_Format was skipped: variadic function \(pyerrors\.h:\d+\)"):
        getattr(h, "PyErr_Format")


def test_type_objects(h):
    tp = ctypes.cast(id(int), ctypes.POINTER(h.PyTypeObject)).contents
    assert tp.tp_name == b"int"
    assert (tp.tp_basicsize, tp.tp_itemsize, tp.tp_flags) == (int.__basicsize__, int.__itemsize__, int.__flags__)
    assert ctypes.sizeof(h.PyObject) == object.__basicsize__
    assert ctypes.cast(id(list), ctypes.POINTER(h.PyTypeObject)).contents.tp_basicsize == list.__basicsize__


def test_objects(h):
    lst = [10, "x", None]
    ob = ctypes.cast(id(lst), ctypes.POINTER(h.PyListObject)).contents
    assert ob.ob_base.ob_size == 3 and ctypes.cast(ob.ob_item[1], ctypes.c_void_p).value == id(lst[1])
    assert ctypes.cast(id(1.5), ctypes.POINTER(h.PyFloatObject)).contents.ob_fval == 1.5
    s = "".join(["ab", "c"])
    ascii_ = ctypes.cast(id(s), ctypes.POINTER(h.PyASCIIObject)).contents
    assert ascii_.length == 3 and ascii_.state.ascii == 1 and ascii_.state.compact == 1
    t = (1, 2)
    assert ctypes.cast(id(t), ctypes.POINTER(h.PyTupleObject)).contents.ob_base.ob_size == 2


def test_constants(h):
    assert h.PY_VERSION_HEX == sys.hexversion
    assert h.PY_VERSION.startswith(f"{sys.version_info.major}.{sys.version_info.minor}")
    assert h.Py_TPFLAGS_HEAPTYPE == 1 << 9
    assert (h.Py_LT, h.Py_GE) == (0, 5)
    assert h.PY_SSIZE_T_MAX == sys.maxsize
    assert h.PyUnicode_1BYTE_KIND == 1
    assert "Py_None" not in vars(h)


def test_opaque_types(h):
    assert not hasattr(h.PyInterpreterState, "_fields_")
    assert h.PyInterpreterState.__name__ == "struct__is"


@pytest.mark.skipif(
    platform.machine().lower() not in ("x86_64", "amd64") or sys.platform == "win32",
    reason="executes x64 code on a POSIX host",
)
def test_generated_code_reads_ob_size(h):
    from jita.x64 import mov, rax, rdi, ret, typed

    @function(ctypes.c_ssize_t, ctypes.c_void_p)
    def py_len():
        mov(rax, typed(rdi, h.PyVarObject).ob_size)
        ret()

    assert py_len(id([1, 2, 3])) == 3
    assert py_len(id((1, 2, 3, 4))) == 4


def test_python_defines(h):
    defines = python_defines()
    assert set(defines) <= {"Py_GIL_DISABLED", "Py_DEBUG", "Py_TRACE_REFS", "Py_STATS"}
    assert all(v == 1 for v in defines.values())
    assert ("Py_GIL_DISABLED" in defines) == bool(sysconfig.get_config_var("Py_GIL_DISABLED"))
    with_defines = load("Python.h", include_dirs=[INCLUDE], defines=defines)
    assert ctypes.sizeof(with_defines.PyObject) == ctypes.sizeof(h.PyObject) == object.__basicsize__
    # pyconfig.h defining the same macros again is neither a skip nor noise.
    assert [str(d) for d in diagnostics(with_defines)] == [str(d) for d in diagnostics(h)]
