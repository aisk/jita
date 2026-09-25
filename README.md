# jita

jita is a Python library for writing x64 machine code by hand and running it
at runtime, at the abstraction level of LuaJIT's
[DynASM](https://luajit.org/dynasm.html). You write instructions, labels and
data directives as ordinary Python calls; jita encodes them, links labels
and external symbols, and loads the result into executable memory that you
can call through `ctypes`. The instruction templates are a port of DynASM's
x86 table (`dasm_x86.lua`, MIT license, Copyright (C) Mike Pall), so
encodings match what DynASM would emit, apart from the differences listed
below. jita has no runtime dependencies.

## Install and run

jita needs Python 3.14 and is developed with [uv](https://docs.astral.sh/uv/):

```sh
uv sync
uv run pytest
uv run python examples/sum_array.py
```

The `examples/` directory has runnable programs: a loop with a macro, a
bytecode interpreter with a dispatch table, calls into libc, an SSE2 dot
product, code specialized by Python-level parameters and a function
assembled from fragments with holes.

## Example

```python
import ctypes
from jita import Assembler
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

## Cheat sheet

| What | jita | DynASM |
| --- | --- | --- |
| registers | `rax`, `r8d`, `al`, `xmm3`, `ymm0`, `st1`, `gp64(n)` | `rax`, `Rq(n)` |
| memory | `qword[rbx + rcx*8 + 8]` | `qword [rbx+rcx*8+8]` |
| size from the other operand | `ptr[rbx]`, `mov(rax, ptr[rbx])` | `[rbx]` |
| rip-relative | `qword[rip + lbl]`, `lea(rax, ptr[rip + lbl])` | `[->lbl]` |
| label | `label("done")`, referenced as `jz("done")`, `qword[rip + "tbl"]`, `a.qword("tbl")` | `->done:`, `->done`, `[->tbl]` |
| label object | `lbl = Label()`, `label(lbl)`, `jz(lbl)` | `1:`, `<1`, `>1` |
| indexed (pc) labels | `a.pc[i]`, `label(a.pc[i])` | `=>i` |
| external symbol | `Extern("strlen")`, address given to `load(externs=...)` | `extern strlen` |
| sections | `with a.section("data"): ...`, `a.section("vars", writable=True)` | `.section` |
| data | `a.byte() a.word() a.dword() a.qword(1, lbl, ext) a.bytes(b"..") a.align(16) a.space(n)` | `.byte .dword .qword .align` |
| short branch | `jmp.short(lbl)`, `jz.short(lbl)` | automatic |
| keyword mnemonics | `and_ or_ not_ int_` | `and or not int` |
| encode once, fill in later | `Fragment()` with `Hole.gp64("r")`, `Hole.imm32("k")`, `Hole.label("l")`; `frag.instantiate(r=rbx, k=5, l=lbl)` | `Rq(r)`, runtime `imm`, `=>l` |
| methods instead of the context | `a.mov(rax, 1)`, `a.label("x")`, `a.jmp.short(lbl)` | |

Macros are plain Python functions that emit instructions, and `.if` is a
Python `if` in the generator. `label("name")` defines a label, and the
same string anywhere a label is accepted refers to it, before or after the
definition. Inside a macro use `Label()` objects instead, so that calling
the macro twice does not define the same name twice. Named labels become
symbols of the loaded module (`mod.function(..., entry="name")`,
`mod.address("name")`), and a Label or Extern used as a `mov r64`
immediate or a `qword` value becomes a 64 bit absolute address. `jita.tools.listing.listing(a)` prints the
generated code one instruction per line with its bytes and relocations.

Loaded code is read-execute, and so are sections by default. Data you want
to modify at runtime goes into a section created with `writable=True`,
which is placed on its own pages and stays read-write. Python can update
it through the module, e.g. `mod.write("counter", (5).to_bytes(8, "little"))`.
`a.align(n)` pads code with NOPs and data sections with zero bytes.

`call(Extern(...))` is a rel32 call, so loading fails with `LinkError` when
the target is more than 2GB away from the code. In that case load the
address into a register first: `mov(rax, Extern("f")); call(rax)`.

Scalar SSE instructions need an explicitly sized memory operand, e.g.
`mulsd(xmm0, qword[rip + k])`; `ptr[...]` does not pick the size there.

## Fragments

`gp64(n)` picks a register while Python generates the code, and the
instruction is encoded again every time. A `Fragment` is encoded once with
`Hole` operands and then instantiated as often as needed; an instance copies
the bytes and patches in the registers, immediates and labels, without
running the encoder:

```python
from jita import Assembler, Fragment, Hole, Label
from jita.x64 import *

dst, src, k, exit_ = Hole.gp64("dst"), Hole.gp64("src"), Hole.imm32("k"), Hole.label("exit")

frag = Fragment()
with frag:
    mov(dst, qword[src + 8])
    add(dst, k)
    jz(exit_)
    label("again")            # local: every instance gets its own label
    dec(dst)
    jnz.short("again")

a = Assembler()
with a:
    done = Label()
    frag.instantiate(dst=rbx, src=rdi, k=1, exit=done)
    inst = frag.instantiate(dst=r12, src=rsp, k=-5, exit=done)
    label(done)
    ret()

inst.labels["again"]          # the label this instance's "again" became
len(frag)                     # 23: every instance has the same length
```

Holes are typed: `Hole.gp8/gp16/gp32/gp64`, `Hole.xmm`, `Hole.ymm` for
registers, `Hole.imm8/imm16/imm32/imm64` for immediates and `Hole.label`
for anything a Label is accepted as (branch targets, `[rip + l]`,
`mov r64, l`, data directives). gp64 holes also work as the base or index
of a memory operand. The value for each hole is passed by name to
`instantiate`, which checks it: a register of the right class and size, an
int in the range the instruction accepts, a Label or a label name.

Since an instance must have the same length for any register, an
instruction with a register hole is always encoded in its general form: a
REX prefix is always present (so `ah`..`bh` cannot be combined with holes),
a base register hole always gets a SIB byte and a displacement, a hole in
the r/m or index field uses the 3 byte VEX prefix, and accumulator short
forms are never used. An immediate hole picks the form by its declared
size, not its value: `add(r, Hole.imm8(..))` is the sign-extended imm8 form
and `mov(eax, Hole.imm8(..))` is an error. `examples/fragments.py` builds a
function out of several instances.

## Differences from DynASM

- No silent branch relaxation. `jmp`/`jcc` to a label always use rel32.
  `jmp.short` always uses rel8 and linking fails if the target is too far.
- Immediates are range checked by value. `add rax, 0xffffffff` is an
  error instead of quietly becoming `add rax, -1`. `mov r64, imm` switches
  to the 64 bit `movabs` form when the value does not fit in 32 bits.
- Some instructions DynASM lacks are added: `xadd`, `cmpxchg`,
  `cmpxchg8b`, `cmpxchg16b`, `ud2`, `hlt` and the `movsq`/`cmpsq`/`stosq`/
  `lodsq`/`scasq` string ops. They combine with `lock()` and `rep()`.
- Everything is known when an instruction is encoded, so there is no
  preprocessor, no action list and no separate link step to call by hand.

## Status

x64 only. Tested on Linux. macOS uses the same mmap/mprotect path but is
untested, and executable memory on Windows is not implemented yet. aarch64
is planned next.
