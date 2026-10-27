"""`Module.builtin`, `Assembler.builtin` and the `builtin` decorator: loaded
code called by CPython as a builtin function."""

import ctypes
import gc
import platform
import weakref

import pytest

from jita import (
    METH_FASTCALL,
    METH_KEYWORDS,
    METH_NOARGS,
    METH_O,
    METH_VARARGS,
    Assembler,
    Extern,
    Label,
    LoadError,
    Module,
    builtin,
)
from jita.runtime import loader
from jita.x64 import *  # noqa: F403

pytestmark = pytest.mark.skipif(
    platform.machine().lower() not in ("x86_64", "amd64") or platform.system() == "Windows",
    reason="needs an x86-64 POSIX host",
)


def capi(name):
    return Extern(getattr(ctypes.pythonapi, name))


def tail_call(name):
    mov(rax, capi(name))
    jmp(rax)


@builtin(METH_O)
def ident():
    """Return the argument."""
    mov(rdi, rsi)
    tail_call("Py_NewRef")


def test_meth_o():
    o = object()
    assert ident(o) is o
    with pytest.raises(TypeError, match=r"ident\(\) takes exactly one argument"):
        ident()


def test_names_and_self():
    assert ident.__name__ == ident.__qualname__ == "ident"
    assert ident.__doc__ == "Return the argument."
    assert ident.__module__ == __name__
    assert repr(ident) == "<built-in function ident>"
    mod = ident.__self__
    assert isinstance(mod, Module)
    assert isinstance(mod.assembler, Assembler)


def test_meth_noargs_receives_the_module():
    @builtin(METH_NOARGS)
    def me():
        tail_call("Py_NewRef")

    assert me() is me.__self__
    with pytest.raises(TypeError, match="takes no arguments"):
        me(1)


def test_meth_fastcall():
    @builtin(METH_FASTCALL)
    def count():
        mov(rdi, rdx)
        tail_call("PyLong_FromSsize_t")

    assert count() == 0
    assert count(1, 2, 3) == 3


def test_meth_fastcall_keywords():
    # PyObject *f(self, args, nargsf, kwnames): returns kwnames or None.
    @builtin(METH_FASTCALL | METH_KEYWORDS)
    def kwnames():
        mov(rdi, rcx)
        test(rdi, rdi)
        jnz("have")
        mov(rdi, capi("_Py_NoneStruct"))
        label("have")
        tail_call("Py_NewRef")

    assert kwnames(1) is None
    assert kwnames(1, a=2, b=3) == ("a", "b")


def test_meth_varargs():
    @builtin(METH_VARARGS)
    def args():
        mov(rdi, rsi)
        tail_call("Py_NewRef")

    assert args(1, "x") == (1, "x")
    with pytest.raises(TypeError):
        args(a=1)


def test_meth_varargs_keywords():
    @builtin(METH_VARARGS | METH_KEYWORDS)
    def kwargs():
        mov(rdi, rdx)
        test(rdi, rdi)
        jnz("have")
        mov(rdi, capi("_Py_NoneStruct"))
        label("have")
        tail_call("Py_NewRef")

    assert kwargs() is None
    assert kwargs(a=1) == {"a": 1}


def test_error_set_by_the_code_is_raised():
    # Returns NULL with the error PyLong_AsSsize_t set for a non int.
    @builtin(METH_O)
    def double():
        sub(rsp, 8)
        mov(rdi, rsi)
        mov(rax, capi("PyLong_AsSsize_t"))
        call(rax)
        cmp(rax, -1)
        jne("ok")
        mov(rax, capi("PyErr_Occurred"))
        call(rax)
        test(rax, rax)
        jz("minus_one")
        xor(eax, eax)
        add(rsp, 8)
        ret()
        label("minus_one")
        mov(rax, -1)
        label("ok")
        lea(rdi, ptr[rax + rax])
        mov(rax, capi("PyLong_FromSsize_t"))
        call(rax)
        add(rsp, 8)
        ret()

    assert double(21) == 42
    assert double(-1) == -2
    with pytest.raises(TypeError):
        double("x")


def test_the_gil_is_held():
    @builtin(METH_NOARGS)
    def gil():
        sub(rsp, 8)
        mov(rax, capi("PyGILState_Check"))
        call(rax)
        movsxd(rdi, eax)
        mov(rax, capi("PyBool_FromLong"))
        call(rax)
        add(rsp, 8)
        ret()

    assert gil() is True


def test_module_builtin_entries():
    with Assembler() as a:
        label("first")
        mov(rdi, rsi)
        tail_call("Py_NewRef")
        label("noargs")
        tail_call("Py_NewRef")
    mod = a.load()
    first = mod.builtin(METH_O, entry="first")
    other = mod.builtin(METH_NOARGS, entry="noargs", name="me", doc="d")
    assert first.__name__ == "first" and first.__module__ is None
    assert other.__name__ == "me" and other.__doc__ == "d"
    assert first(5) == 5 and other() is mod
    anon = mod.builtin(METH_O)
    assert anon.__name__ == "builtin" and anon(7) == 7


def test_assembler_builtin():
    with Assembler() as a:
        mov(rdi, rsi)
        tail_call("Py_NewRef")
    fn = a.builtin(METH_O, name="same", module_name="m")
    assert fn(3) == 3 and fn.__name__ == "same" and fn.__module__ == "m"
    assert fn.__self__.assembler is a


@pytest.mark.parametrize("flags", [True, 8.0, 0, METH_KEYWORDS, METH_O | METH_KEYWORDS, METH_NOARGS | METH_O, 0x200])
def test_unsupported_flags(flags):
    with Assembler() as a:
        ret()
    with pytest.raises(ValueError, match="calling convention"):
        a.load().builtin(flags)


def test_closed_module():
    with Assembler() as a:
        ret()
    mod = a.load()
    mod.close()
    with pytest.raises(LoadError):
        mod.builtin(METH_O)


def test_function_keeps_the_module_alive():
    @builtin(METH_O)
    def same():
        mov(rdi, rsi)
        tail_call("Py_NewRef")

    ref = weakref.ref(same.__self__)
    gc.collect()
    assert ref() is not None and same(1) == 1
    del same
    gc.collect()
    assert ref() is None


def test_label_entries():
    named, anonymous = Label("named"), Label()
    with Assembler() as a:
        label(named)
        label(anonymous)
        mov(rdi, rsi)
        tail_call("Py_NewRef")
    mod = a.load()
    assert mod.builtin(METH_O, entry=named).__name__ == "named"
    assert mod.builtin(METH_O, entry=anonymous).__name__ == "builtin"


def test_decorator_entry_and_externs():
    @builtin(METH_O, entry="real", externs={"newref": ctypes.cast(ctypes.pythonapi.Py_NewRef, ctypes.c_void_p).value})
    def f():
        ud2()
        label("real")
        mov(rdi, rsi)
        jmp(qword[rip + Extern("newref")])

    assert f(9) == 9 and f.__name__ == "f"


def test_method_defs_are_shared_by_content():
    with Assembler() as a:
        mov(rdi, rsi)
        tail_call("Py_NewRef")
    mod = a.load()
    before = len(loader._method_defs)
    mod.builtin(METH_O, name="x")
    mod.builtin(METH_O, name="x")
    assert len(loader._method_defs) == before + 1
    mod.builtin(METH_O, name="x", doc="d")
    assert len(loader._method_defs) == before + 2


def test_module_dir_lists_methods():
    assert {"builtin", "function", "write", "close", "assembler"} <= set(dir(ident.__self__))
    assert ident.__self__.__doc__ == Module.__doc__


def test_method_def_outlives_a_collected_cycle():
    # The cyclic GC clears the module dict before the builtin is freed, and
    # CPython reads the PyMethodDef while freeing the builtin, so the def
    # must not be owned by anything the module holds.
    with Assembler() as a:
        mov(rax, rsi)
        ret()

    class C:
        f = a.builtin(METH_O, name="in_cycle")

    mod = C.f.__self__
    md = loader._method_defs[(mod.image.base, METH_O, "in_cycle", None)]
    module_ref = weakref.ref(mod)
    del C, a, mod
    gc.collect()
    assert module_ref() is None
    assert any(kept is md for kept in loader._method_defs.values())
    assert md.ml_flags == METH_O and md.ml_name == b"in_cycle"
