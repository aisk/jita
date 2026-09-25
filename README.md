# jita

jita is a Python library for writing x64 and aarch64 machine code by hand
and running it at runtime, at the abstraction level of LuaJIT's
[DynASM](https://luajit.org/dynasm.html). You write instructions, labels and
data directives as ordinary Python calls; jita encodes them, links labels
and external symbols, and loads the result into executable memory that you
can call through `ctypes`. The instruction templates are ports of DynASM's
x86 and ARM64 tables (`dasm_x86.lua` and `dasm_arm64.lua`, MIT license,
Copyright (C) Mike Pall), so encodings match what DynASM would emit, apart
from the differences listed below. jita has no runtime dependencies.

## Install and run

jita needs Python 3.14 and is developed with [uv](https://docs.astral.sh/uv/):

```sh
uv sync
uv run pytest
uv run python examples/sum_array.py
```

The `examples/` directory has runnable programs: a loop with a macro, a
bytecode interpreter with a dispatch table, calls into libc, an SSE2 dot
product, a linked list of ctypes structures, code specialized by
Python-level parameters, a function assembled from fragments with holes
and an aarch64 function whose listing prints on any host.

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
| call through a pointer slot | `call(qword[rip + Extern("strlen")])` | |
| structure fields | `p = typed(rdi, Point)`, `mov(eax, p.x)`, `p.v[rcx]` | `.type P, Point, rdi`, `P->x` |
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
the target is more than 2GB away from the code, which is common for shared
libraries. The portable form is `call(qword[rip + ext])`: an Extern used as
a rip-relative memory operand refers to an 8 byte slot holding the extern's
address. jita creates one slot per extern name in a read-only `externs`
section, which is laid out next to the code, so the same slot serves
`call(qword[rip + ext])`, `jmp(qword[rip + ext])` (a tail call),
`mov(rax, qword[rip + ext])` and `lea(rax, ptr[rip + ext])` (the slot's own
address). `mov(rax, ext); call(rax)` also works at any distance.

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
int in the range the instruction accepts, a Label, a label name or an
Extern. For an Extern, `jmp(l)` and `mov(rax, l)` refer to the extern
itself and `[rip + l]` to its pointer slot. A fragment can be instantiated
into another fragment, and a hole of the outer fragment is a valid value
for an inner hole of the same type, which leaves that field open until the
outer fragment is instantiated.

Since an instance must have the same length for any register, an
instruction with a register hole is always encoded in its general form: a
REX prefix is always present (so `ah`..`bh` cannot be combined with holes),
a base register hole always gets a SIB byte and a displacement, a hole in
the r/m or index field uses the 3 byte VEX prefix, and accumulator short
forms are never used. An immediate hole picks the form by its declared
size, not its value: `add(r, Hole.imm8(..))` is the sign-extended imm8 form
and `mov(eax, Hole.imm8(..))` is an error. `examples/fragments.py` builds a
function out of several instances.

## Structures

`typed(base, Type)` views memory as a `ctypes.Structure` or `Union`, so
generated code and Python share one definition of the layout:

```python
import ctypes
from jita import Assembler
from jita.x64 import *

class Point(ctypes.Structure):
    _fields_ = [("x", ctypes.c_int32), ("y", ctypes.c_int32), ("tag", ctypes.c_uint8),
                ("next", ctypes.c_void_p), ("v", ctypes.c_double * 4)]

p = typed(rdi, Point)             # base register, or an address like rdi + 16
with Assembler():
    mov(eax, p.x)                 # dword[rdi]
    movzx(eax, p.tag)             # byte[rdi+8]
    movsd(xmm0, p.v[1])           # qword[rdi+32]
    movsd(xmm1, p.v[rcx])         # qword[rdi+rcx*8+24]
    lea(rax, p.v.addr)            # [rdi+24], unsized
    mov(rdi, p.next)              # qword[rdi+16]
```

Scalar fields are sized memory operands (1, 2, 4 or 8 bytes, pointers are
qwords), nested structures and unions are further views and arrays take a
constant or a register index. Offsets are the ones ctypes computes, so
`_pack_`, `_anonymous_` and unions behave as in Python. `p.addr`, `p.size`
and `p.ctype` give the start address, `ctypes.sizeof` and the type; a field
that clashes with those names is reached as `p["size"]`. Pointer fields are
not followed: load the pointer and call `typed` on the register again. Bit
fields and `c_longdouble` have no memory operand and raise `EncodeError`.
In a fragment, gp64 register holes work as the base and as an array index:
`typed(Hole.gp64("s"), Point).v[Hole.gp64("i")]`.

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

## aarch64

`jita.aarch64` works like `jita.x64`. `Assembler()` picks it on an aarch64
host; `Assembler(jita.aarch64)` generates aarch64 code anywhere, for
example to print a listing.

```python
from jita import Assembler
import jita.aarch64 as arm
from jita.aarch64 import *

a = Assembler(arm)
with a:                           # int64_t sum(int64_t *p, size_t n)
    mov(x2, xzr)
    cbz(x1, "done")
    label("loop")
    ldr(x3, mem.post[x0, 8])
    add(x2, x2, x3)
    subs(x1, x1, 1)
    b.ne("loop")
    label("done")
    mov(x0, x2)
    ret()
```

Registers are `x0`..`x30`, `w0`..`w30`, `xzr`, `wzr`, `sp`, the FP
registers `s0`..`s31` and `d0`..`d31`, and `q0`..`q31` for `ldp`/`stp`.
`lr` and `fp` are `x30` and `x29`. `gp64(n)`, `gp32(n)`, `fp32(n)`,
`fp64(n)` and `fp128(n)` select a register by number (31 is xzr/wzr).
Access sizes come from the register and the mnemonic (`ldrb`, `ldrsh`,
`ldr w0`, `ldr d0`), so memory operands have no size prefix.

| What | jita | DynASM / GNU as |
| --- | --- | --- |
| shifted register | `x2 << 3`, `x2 >> 3`, `x2.lsl(3)`, `x2.lsr(3)`, `x2.asr(3)` | `x2, lsl #3`, `x2, lsr #3`, `x2, asr #3` |
| extended register | `w2.uxtw()`, `w2.sxtw(2)`, `x2.sxtx(1)`, also `uxtb uxth uxtx sxtb sxth` | `w2, uxtw`, `w2, sxtw #2` |
| immediate | `add(x0, x1, 16)`, `fmov(d0, 1.5)` | `#16`, `#1.5` |
| wide move shift | `movk(x0, 0xbeef, lsl=16)` | `movk x0, #0xbeef, lsl #16` |
| base, offset | `mem[x0]`, `mem[x0 + 8]`, `mem[sp - 16]` | `[x0]`, `[x0, #8]`, `[sp, #-16]` |
| register offset | `mem[x0 + x1]`, `mem[x0 + (x1 << 3)]`, `mem[x0 + w1.uxtw(2)]` | `[x0, x1]`, `[x0, x1, lsl #3]`, `[x0, w1, uxtw #2]` |
| pre-index | `mem.pre[sp - 16]` | `[sp, #-16]!` |
| post-index | `mem.post[x0, 8]` | `[x0], #8` |
| condition | `csel(x0, x1, x2, "ne")`, `cset(x0, "lo")` | `ne`, `lo` |
| conditional branch | `b.eq(lbl)` or `beq(lbl)` | `beq lbl` / `b.eq lbl` |
| label, literal | `b("loop")`, `adr(x0, lbl)`, `ldr(x0, "const")` | `b ->loop`, `adr x0, ->lbl` |
| keyword mnemonics | `and_`, `str_` (also `a.str(...)`) | `and`, `str` |

Condition codes are `eq ne cs hs cc lo mi pl vs vc hi ls ge lt gt le al`.
The offset form of a load or store is chosen from the value: a scaled
unsigned offset when it fits (`ldr x0, [x1, #8]`), otherwise a signed
9 bit unscaled one (`ldur`). Parenthesize a shifted index,
`mem[x0 + (x1 << 3)]`, since `<<` binds weaker than `+`.

The instruction set is DynASM's ARM64 table: integer arithmetic and
logic with shifted, extended and immediate operands, moves (`mov`,
`movz`, `movn`, `movk`), conditional select and compare, multiply and
divide, bitfield operations and their aliases, loads and stores of
every size with all addressing modes, load/store pair, branches
(`b`, `bl`, `br`, `blr`, `ret`, `cbz`, `tbz`, `b.cond`), `adr`, `adrp`,
`bti`, pointer authentication branches, `nop`, `brk`, and scalar
single/double floating point (arithmetic, `fmadd` family, conversions,
rounding, compares, `fcsel`, `fmov` with immediates). SIMD, atomics,
system instructions and barriers are not in the table. `mov` accepts the
immediates that `movz` or a logical immediate can encode; build other
constants with `movz`/`movk`.

jita checks operands more strictly than a plain field encoder: register
31 is only accepted as `sp` where the instruction means the stack
pointer and only as `xzr` where it means zero, 32 bit instructions reject
shift amounts and bit positions above 31, bitfield aliases check lsb and
width, the `cset`/`cinc` family rejects `al`, and loads that write back
into a register they also load are errors. Branches to labels are linked
with `target - instruction address`; `adrp` uses the 4KB page
difference.

## Status

x64 and aarch64. Tested on Linux. aarch64 encodings are checked against
DynASM and an aarch64 assembler on an x86 host; running aarch64 code has
not been tried on hardware yet. macOS uses the same mmap/mprotect path but
is untested, and executable memory on Windows is not implemented yet.
