# jita

jita is a Python library for writing x64, aarch64, riscv64 and loongarch64
machine code by hand and running it at runtime, in the spirit of LuaJIT's
[DynASM](https://luajit.org/dynasm.html). Instructions, labels and data
directives are ordinary Python calls; jita encodes them, links labels and
external symbols, and loads the result into executable memory that you
call through `ctypes`. It has no runtime dependencies and needs Python 3.14.
The API ships type stubs, so pyright and mypy check instruction operands.

```python
import ctypes, sysconfig
from jita import Extern, function
from jita.cheader import load, python_defines
from jita.x64 import *

# The interpreter's own Python.h, turned into ctypes types at runtime.
h = load("Python.h", include_dirs=[sysconfig.get_path("include")], defines=python_defines())
PyTupleObject = h.PyTupleObject
size = Extern(ctypes.pythonapi.PyObject_Size)

@function(ctypes.c_ssize_t, ctypes.py_object, functype=ctypes.PYFUNCTYPE)
def total_len():                           # sum(map(len, t)) for a tuple t
    push(rbx); push(r12); push(r13)        # callee saved, aligns rsp to 16
    mov(rbx, rdi)
    t = typed(rbx, PyTupleObject)          # offsets come from Python.h
    xor(r12d, r12d)                        # i
    xor(r13d, r13d)                        # total
    label("loop")
    cmp(r12, t.ob_base.ob_size)
    jge("done")
    mov(rdi, t.ob_item[r12])
    call(qword[rip + size])                # PyObject_Size(t[i])
    test(rax, rax)
    js("done")                             # -1: error set, ctypes raises it
    add(r13, rax)
    inc(r12)
    jmp("loop")
    label("done")
    mov(rax, r13)
    pop(r13); pop(r12); pop(rbx)
    ret()

print(total_len(("ab", [1, 2, 3], {})))    # 5
try:
    total_len(("ab", 1))
except TypeError as e:
    print(e)                               # ... has no len()
```

`jita.cheader.load` reads C headers and builds `ctypes` structures from
them (`pip install "jita[cheader]"`), and `typed()` views memory through
such a structure, so the generated code reads CPython objects with the
layout of the running build. `Extern` names a symbol for the code to call,
here a C API function. `@function` runs the body once inside a fresh
`Assembler`, loads the result and replaces the name with the `ctypes`
callable. Like any ctypes call it releases the GIL by default;
`ctypes.PYFUNCTYPE` keeps it, which the C API needs, and raises the Python
error set by the code. The body is a code generator: inside a factory it
closes over the factory's parameters, which become immediates, and Python
`if` statements can decide what code is emitted.
Registers are objects (`rax`, `r8d`, `xmm0`, `x0`, `w1`), memory operands
are written as `qword[rbx + rcx*8 + 8]` or `mem[x0 + 8]`, labels are
strings or `Label` objects, and macros are plain Python functions. Without
the decorator, `a.function(...)` turns any `Assembler` into a callable,
and `a.load()` gives a `Module` for code with several entry points or
writable data. Beyond that jita has sections, `Extern` symbols and a
listing tool that prints the generated code with its bytes.

The full API is described in [docs/reference.md](docs/reference.md). The
`examples/` directory has runnable programs: a loop with a macro, a
bytecode interpreter with a dispatch table, three ways to call into libc
from one module, an SSE2 dot product, a linked list of ctypes structures,
a factory that specializes code by Python parameters, CPython objects
read with the layouts of Python.h, and aarch64, riscv64 and loongarch64
functions whose listings print on any host.

## Install and run

```sh
uv add jita                        # in your project, or pip install jita
uv add "jita[cheader]"             # with jita.cheader, as in the example above
uv sync && uv run pytest           # working on jita itself
uv run python examples/sum_array.py
```

## Status

x64 and aarch64 on Linux, macOS and Windows.
riscv64 (RV64 with I, M, A, F, D, Zicsr, Zifencei, Zba and Zbb) on Linux.
loongarch64 (LA64 base integer, atomics and scalar floating point) on Linux.

## License

MIT. The x64 and aarch64 instruction tables are derived from DynASM, see
LICENSE.
