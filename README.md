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

@function(ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_size_t)
def sum_array():                  # the body is a generator that runs once
    xor(eax, eax)                 # int64_t sum_array(int64_t *p, size_t n)
    test(rsi, rsi)
    jz("done")
    label("loop")
    add(rax, qword[rdi])
    add(rdi, 8)
    dec(rsi)
    jnz("loop")
    label("done")
    ret()

print(sum_array((ctypes.c_int64 * 3)(1, 2, 3), 3))   # 6
```

`@function` runs the body inside a fresh `Assembler`, loads the result
and replaces the name with the `ctypes` callable. Inside a factory the body
closes over its parameters, so Python values and `if` statements decide
what code is emitted. Registers are objects (`rax`, `r8d`, `xmm0`, `x0`,
`w1`), memory operands are written as `qword[rbx + rcx*8 + 8]` or
`mem[x0 + 8]`, labels are strings or `Label` objects, and macros are plain
Python functions. Without the decorator, `a.function(...)` turns any
`Assembler` into a callable, and `a.load()` gives a `Module` for code with
several entry points or writable data. Beyond that jita has sections,
`Extern` symbols, `typed()` views over `ctypes` structures, and a listing
tool that prints the generated code with its bytes.

The full API is described in [docs/reference.md](docs/reference.md). The
`examples/` directory has runnable programs: a loop with a macro, a
bytecode interpreter with a dispatch table, three ways to call into libc
from one module, an SSE2 dot product, a linked list of ctypes structures,
a factory that specializes code by Python parameters and an aarch64
function whose listing prints on any host.

## Install and run

```sh
uv add jita                        # in your project
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
