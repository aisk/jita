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
from jita import function
from jita.cheader import load, python_defines
from jita.x64 import *

# The interpreter's own Python.h, turned into ctypes types at runtime.
h = load("Python.h", include_dirs=[sysconfig.get_path("include")], defines=python_defines())

@function(ctypes.c_double, ctypes.py_object)
def fsum():                                # sum of a list of floats
    xs = typed(rdi, h.PyListObject)        # field offsets come from Python.h
    mov(rcx, xs.ob_base.ob_size)           # Py_SIZE(xs)
    mov(rsi, xs.ob_item)
    xorpd(xmm0, xmm0)
    test(rcx, rcx)
    jz("done")
    label("loop")
    mov(rax, qword[rsi])                   # PyObject *item
    addsd(xmm0, typed(rax, h.PyFloatObject).ob_fval)
    add(rsi, 8)
    dec(rcx)
    jnz("loop")
    label("done")
    ret()

print(fsum([1.5, 2.0, 3.25]))              # 6.75
```

`jita.cheader.load` reads C headers and builds `ctypes` structures from
them (`pip install "jita[cheader]"`), and `typed()` views memory through
such a structure, so the generated code reads CPython objects with the
layout of the running build. `@function` runs the body once inside a
fresh `Assembler`, loads the result and replaces the name with the
`ctypes` callable. The body is a code generator: inside a factory it
closes over the factory's parameters, which become immediates, and
Python `if` statements can decide what code is emitted.
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
