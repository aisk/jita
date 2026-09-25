# jita

jita is a Python library for writing x64 and aarch64 machine code by hand
and running it at runtime, in the spirit of LuaJIT's
[DynASM](https://luajit.org/dynasm.html). Instructions, labels and data
directives are ordinary Python calls; jita encodes them, links labels and
external symbols, and loads the result into executable memory that you
call through `ctypes`. It has no runtime dependencies and needs Python 3.14.

```python
import ctypes
from jita import Assembler, label
from jita.x64 import *

def build():
    a = Assembler()
    with a:                       # module level mnemonics emit into `a`
        xor(eax, eax)             # int64_t sum(int64_t *p, size_t n)
        test(rsi, rsi)
        jz("done")
        label("loop")
        add(rax, qword[rdi])
        add(rdi, 8)
        dec(rsi)
        jnz("loop")
        label("done")
        ret()
    return a

with build().load() as mod:
    fn = mod.function(ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_size_t)
    print(fn((ctypes.c_int64 * 3)(1, 2, 3), 3))   # 6
```

Registers are objects (`rax`, `r8d`, `xmm0`, `x0`, `w1`), memory operands
are written as `qword[rbx + rcx*8 + 8]` or `mem[x0 + 8]`, labels are
strings or `Label` objects, and macros are plain Python functions. Beyond
that jita has sections and writable data, `Extern` symbols, `Fragment`s that
are encoded once and instantiated with different registers, `typed()` views
over `ctypes` structures, and a listing tool that prints the generated code
with its bytes.

The full API is described in [docs/reference.md](docs/reference.md). The
`examples/` directory has runnable programs: a loop with a macro, a
bytecode interpreter with a dispatch table, calls into libc, an SSE2 dot
product, a linked list of ctypes structures, code specialized by
Python-level parameters, a function assembled from fragments and an
aarch64 function whose listing prints on any host.

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
