# jita

jita is a Python library for writing x64 and aarch64 machine code by hand
and running it at runtime, in the spirit of LuaJIT's
[DynASM](https://luajit.org/dynasm.html). Instructions, labels and data
directives are ordinary Python calls; jita encodes them, links labels and
external symbols, and loads the result into executable memory that you
call through `ctypes`. It has no runtime dependencies and needs Python 3.14.

```python
import ctypes
from jita import function
from jita.x64 import *

class Vec(ctypes.Structure):
    _fields_ = [("data", ctypes.POINTER(ctypes.c_int64)), ("len", ctypes.c_size_t)]

def make_scaled_sum(k):               # int64_t f(Vec *v): k * sum(v->data[0..len))
    @function(ctypes.c_int64, ctypes.POINTER(Vec))
    def f():
        v = typed(rdi, Vec)           # field offsets come from ctypes
        mov(rsi, v.data)              # qword[rdi]
        mov(rcx, v.len)               # qword[rdi + 8]
        xor(eax, eax)
        test(rcx, rcx)
        jz("done")
        label("loop")
        add(rax, qword[rsi])
        add(rsi, 8)
        dec(rcx)
        jnz("loop")
        label("done")
        imul(rax, rax, k)             # k is an immediate in the generated code
        ret()
    return f

arr = (ctypes.c_int64 * 3)(1, 2, 3)
print(make_scaled_sum(10)(Vec(arr, 3)))   # 60
```

`@function` runs the body once inside a fresh `Assembler`, loads the
result and replaces the name with the `ctypes` callable. The body is a
code generator: inside a factory it closes over the factory's parameters,
so `k` above becomes an immediate and Python `if` statements can decide
what code is emitted. `typed()` views memory through a `ctypes` structure,
so generated code and Python share one definition of the layout.
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
a factory that specializes code by Python parameters and an aarch64
function whose listing prints on any host.

## Install and run

```sh
uv add jita                        # in your project, or pip install jita
uv sync && uv run pytest           # working on jita itself
uv run python examples/sum_array.py
```

## Status

x64 and aarch64. Tested on Linux. aarch64 encodings are checked against
DynASM and an aarch64 assembler on an x86 host; running aarch64 code has
not been tried on hardware yet. macOS uses the same mmap/mprotect path but
is untested, and executable memory on Windows is not implemented yet.

## License

MIT. The instruction tables are derived from DynASM, see LICENSE.
