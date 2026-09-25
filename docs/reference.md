# jita reference

jita generates machine code at runtime from ordinary Python calls, the way
DynASM does from a preprocessed C file. This page covers the whole API.
The README has the short version.

Contents: [Assembler](#assembler) · [Sections](#sections) ·
[Labels](#labels) · [Externs](#externs) · [Data](#data-directives) ·
[Linking and loading](#linking-and-loading) · [Listings](#listings) ·
[x64](#x64) · [Fragments](#fragments) · [Structures](#structures) ·
[aarch64](#aarch64) · [Compared with DynASM](#compared-with-dynasm) ·
[Errors](#errors)

## Assembler

```python
from jita import Assembler, label
from jita.x64 import *

a = Assembler()               # host architecture
a = Assembler("x64")          # or "aarch64", or the package jita.x64
with a:                       # module level mnemonics emit into `a`
    mov(rax, 1)
    ret()
a.mov(rax, 1)                 # every mnemonic is also a method
```

`with a:` makes `a` the current assembler for the module level instruction
functions, the `label()` directive and `Fragment.instantiate()`. Contexts
nest, so a macro is just a Python function that emits instructions, and
DynASM's `.if` is a Python `if` in the generator. `jita.current()` returns
the active assembler. Outside any context use the method forms.

Instruction names are lowercase. The x64 mnemonics that clash with Python
keywords or builtins are `and_`, `or_`, `not_` and `int_`; on aarch64
`and_` and `str_` (the methods `a.and_` and `a.str` both exist).
`from jita.x64 import *` exports registers, size prefixes, `label`,
`typed` and the mnemonics, and nothing else. Note that it exports `test`,
which pytest collects as a test; write `del test` or import the module
qualified in test files.

## Sections

Code and data go into named sections. The default section is `code`.

```py
with a.section("data"):           # switch for the block
    a.align(8)
    label("table")
    a.qword(1, 2, 3)
a.section("vars", writable=True)  # switch permanently
```

`section(name, align=None, writable=None)` creates the section on first
use. `align` raises the section's alignment in the linked image (`a.align(n)`
inside a section does the same). Sections are laid out in creation order,
read-only sections first, then writable sections starting on a fresh page.
Loaded code and read-only data are mapped read-execute; a section created
with `writable=True` stays read-write so Python and the generated code can
update it (see `Module.write`). `a.align(n)` pads code sections with NOPs
and data sections with zero bytes; `a.align(n, fill=b"..")` sets the fill.

## Labels

```py
jz("done")                # reference by name, before or after the definition
label("done")             # define at the current position
lbl = Label()             # anonymous label object
label(lbl); jnz(lbl)
label(a.pc[3])            # indexed labels, DynASM's =>n
lbl = a.label("entry")    # method form, returns the Label
```

A string names a label of the assembler wherever a Label is accepted:
branch targets, `[rip + "tbl"]`, `mov(rax, "tbl")`, data directives. The
same name always means the same label, `a.named("x")` returns it, and
linking fails if a referenced name is never defined. Inside a macro use
`Label()` objects, so that calling the macro twice does not define the same
name twice. `a.pc[i]` returns the same anonymous label for the same index,
like DynASM's `=>i`; `len(a.pc)` is the highest index plus one.

Named labels become symbols of the linked image and the loaded module
(`img.address("entry")`, `mod.function(..., entry="entry")`). A label used
as a `mov r64` immediate or as a `qword` value becomes its 64 bit absolute
address. Labels are bound once; binding twice or using a label bound in
another assembler is a `LinkError`.

Branches to labels are never relaxed: `jmp`/`jcc` always use rel32, and
`jmp.short(lbl)`, `jz.short(lbl)` always use rel8 and fail at link time
when the target is out of range.

## Externs

```py
from jita import Extern
strlen = Extern("strlen")             # address supplied at load time
puts = Extern("puts", 0x7f00_1234)    # or fixed up front
call(qword[rip + strlen])             # call through the pointer slot
mov(rax, strlen); call(rax)           # 64 bit absolute address
call(strlen)                          # direct rel32, must be within 2GB
mod = a.load(externs={"strlen": addr})
```

The `externs` mapping given to `link` or `load` wins over an address fixed
in the constructor. `call(ext)` and `jmp(ext)` are rel32 and fail with
`LinkError` when the target is more than 2GB from the code, which is common
for shared libraries. An Extern used as a rip-relative memory operand
refers instead to an 8 byte slot holding the extern's address. jita
creates one slot per extern name in a read-only `externs` section laid
out next to the code (`a.extern_slot(ext)` returns its label), so
`call(qword[rip + ext])`, `jmp(qword[rip + ext])`,
`mov(rax, qword[rip + ext])` and `lea(rax, ptr[rip + ext])` work at any
distance. A displacement on a slot operand is an error.

## Data directives

```py
a.byte(1, -1)                 # 1 byte each, signed or unsigned range
a.word(0x1234)
a.dword(lbl)                  # 32 bit absolute address of a label
a.qword(1, "tbl", strlen)     # ints, label names, Labels and Externs
a.bytes(b"\x90\x90")
a.space(16)                   # 16 zero bytes, a.space(16, 0xcc) to fill
a.align(16)
```

Values are range checked by size. Labels and Externs become absolute
addresses of the directive's width (`ABS8`..`ABS64` patches), so a `dword`
of a label only links when the image base fits in 32 bits.

## Linking and loading

```py
img = a.link(base=0x1000, externs={...})   # Image: bytes plus addresses
img.data, img.address("entry"), img.section_offsets, img.symbols

with a.load(externs={...}) as mod:         # Module: mapped and executable
    fn = mod.function(ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_size_t)
    fn = mod.function(ctypes.c_int, entry="second")
    mod.address("table")
    mod.write("counter", (5).to_bytes(8, "little"))   # writable sections only
```

`link` resolves every label and extern patch for the given base and
returns an `Image`. `load` links at the address of freshly mapped memory,
copies the image in, marks the read-only prefix executable and returns a
`Module`. `Module.function(restype, *argtypes, entry=None)` builds a
`ctypes` function pointer at `entry` (default: the start of the image); the
function keeps the module alive. `Module.write` updates bytes in writable
sections; writing into the executable prefix is a `LoadError`. `close()` or
leaving the `with` block unmaps the memory. An assembler can be linked or
loaded more than once, and labels bound after linking are detected.

## Listings

```py
from jita.tools.listing import listing
print(listing(a))             # offsets, bytes, one instruction per line
print(listing(a, img))        # with linked addresses
```

Listings show every section, the bound labels, the bytes of each
instruction with its text, and relocation notes such as `rel32 -> done`.
Anonymous labels are numbered `.L1`, `.L2` in bind order, pc labels print
as `.Lpc3`, extern slots as `strlen@slot`.

## x64

### Registers and operands

| Operand | jita | DynASM |
| --- | --- | --- |
| registers | `rax`, `r8d`, `al`, `ah`, `ax`, `xmm3`, `ymm0`, `st1` | `rax`, `Rq(n)` |
| by number | `gp8(n) gp16(n) gp32(n) gp64(n) xmm(n) ymm(n) st(n)` | `Rb(n) Rw(n) Rd(n) Rq(n)` |
| memory | `qword[rbx + rcx*8 + 8]`, `dword[rax]`, `byte[rip + lbl]` | `qword [rbx+rcx*8+8]` |
| size from the other operand | `ptr[rbx]` as in `mov(rax, ptr[rbx])` | `[rbx]` |
| sizes | `byte word dword qword oword yword tword ptr` | `byte word dword qword oword yword` |
| rip-relative | `qword[rip + lbl]`, `lea(rax, ptr[rip + lbl])` | `[->lbl]` |
| absolute address | `qword[0x1000]`, `dword[eax]` (with 0x67) | `[0x1000]` |
| immediates | plain ints, range checked by value | `imm` |
| prefixes | `lock(); add(qword[rdi], 1)`, `rep(); movsq()` | `lock; add ...` |
| short branch | `jmp.short(lbl)`, `jz.short(lbl)` | automatic |

Memory operands are built with operator overloading: `base + index*scale +
disp`, a label only with a `rip` base, a bare integer as an absolute
address. Scalar SSE instructions need an explicitly sized memory operand
(`mulsd(xmm0, qword[rip + k])`); `ptr[...]` does not pick the size there.
`ah`, `bh`, `ch`, `dh` cannot be combined with anything that needs a REX
prefix (r8..r15, spl..dil, 64 bit operands).

### Immediates and encoding choices

`mov(rax, imm)` uses the 32 bit sign extended form when the value fits and
the 64 bit `movabs` form otherwise; `movabs`/`mov64` request it. Other
instructions check the mathematical value against the operand size, so
`add(rax, 0xffffffff)` is an error rather than `add rax, -1`. `xchg(eax,
eax)` is encoded as `87 C0`, not as a NOP, since it zero-extends.

Beyond DynASM's table jita adds `xadd`, `cmpxchg`, `cmpxchg8b`,
`cmpxchg16b`, `ud2`, `hlt` and the 64 bit string operations `movsq`,
`cmpsq`, `stosq`, `lodsq`, `scasq`; they combine with `lock()` and `rep()`.
16 bit `lea` is not available.

## Fragments

`gp64(n)` picks a register while Python generates the code, and the
instruction is encoded again every time. A `Fragment` is encoded once with
`Hole` operands and then instantiated as often as needed; an instance copies
the bytes and patches in the registers, immediates and labels, without
running the encoder. This is DynASM's `Rq(n)`, runtime `imm` and `=>pc`.

```python
from jita import Assembler, Fragment, Hole, Label, label
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
registers (`Hole.fp32/fp64` on aarch64), `Hole.imm8/imm16/imm32/imm64` for
immediates and `Hole.label` for anything a Label is accepted as (branch
targets, `[rip + l]`, `mov r64, l`, data directives). gp64 holes also work
as the base or index of a memory operand. The value for each hole is passed
by name to `instantiate`, which checks it: a register of the right class and
size, an int in the range the instruction accepts, a Label, a label name or
an Extern. For an Extern, `jmp(l)` and `mov(rax, l)` refer to the extern
itself and `[rip + l]` to its pointer slot. A hole that appears in no
instruction must be announced with `frag.declare(hole)` before it can be
passed. A fragment can be instantiated into another fragment, and a hole of
the outer fragment is a valid value for an inner hole of the same type,
which leaves that field open until the outer fragment is instantiated.

Since an instance must have the same length for any register, an
instruction with a register hole is always encoded in its general form: a
REX prefix is always present (so `ah`..`bh` cannot be combined with holes),
a base register hole always gets a SIB byte and a displacement, a hole in
the r/m or index field uses the 3 byte VEX prefix, and accumulator short
forms are never used. An immediate hole picks the form by its declared
size, not its value: `add(r, Hole.imm8(..))` is the sign-extended imm8 form
and `mov(eax, Hole.imm8(..))` is an error. `Fragment(align=16)` requires
instances to start at that alignment. `examples/fragments.py` builds a
function out of several instances.

## Structures

`typed(base, Type)` views memory as a `ctypes.Structure` or `Union`, so
generated code and Python share one definition of the layout.

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

## aarch64

`jita.aarch64` works like `jita.x64`. `Assembler()` picks it on an aarch64
host; `Assembler("aarch64")` generates aarch64 code anywhere, for example
to print a listing.

```python
from jita import Assembler
from jita.aarch64 import *

a = Assembler("aarch64")
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

| Operand | jita | GNU as |
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
| conditional branch | `b.eq(lbl)` or `beq(lbl)` | `b.eq lbl` |
| label, literal | `b("loop")`, `adr(x0, lbl)`, `ldr(x0, "const")` | `b loop`, `adr x0, lbl` |
| keyword mnemonics | `and_`, `str_` (also `a.str(...)`) | `and`, `str` |

Condition codes are `eq ne cs hs cc lo mi pl vs vc hi ls ge lt gt le al`.
The offset form of a load or store is chosen from the value: a scaled
unsigned offset when it fits (`ldr x0, [x1, #8]`), otherwise a signed
9 bit unscaled one (`ldur`). Parenthesize a shifted index,
`mem[x0 + (x1 << 3)]`, since `<<` binds weaker than `+`.

The instruction set covers integer arithmetic and logic with shifted,
extended and immediate operands, moves (`mov`, `movz`, `movn`, `movk`),
conditional select and compare, multiply and divide, bitfield operations
and their aliases, loads and stores of every size with all addressing
modes, load/store pair, branches (`b`, `bl`, `br`, `blr`, `ret`, `cbz`,
`tbz`, `b.cond`), `adr`, `adrp`, `bti`, pointer authentication branches,
`nop`, `brk`, and scalar single/double floating point (arithmetic, the
`fmadd` family, conversions, rounding, compares, `fcsel`, `fmov` with
immediates). SIMD, atomics, system instructions and barriers are not
covered. `mov` takes a register, an unshifted 16 bit immediate (encoded as
`movz`) or a logical immediate (encoded as `orr`); it never picks a shifted
`movz` or a `movn`, so build other constants with `movz`, `movn` and `movk`
and their `lsl=` keyword (`movz(x0, 0x1234, lsl=16)`).

Operands are checked strictly: register 31 is only accepted as `sp` where
the instruction means the stack pointer and only as `xzr` where it means
zero, 32 bit instructions reject shift amounts and bit positions above 31,
bitfield aliases check lsb and width, the `cset`/`cinc` family rejects
`al`, and loads that write back into a register they also load are errors.
Branches to labels are linked with `target - instruction address`; `adrp`
uses the 4KB page difference. Since there is no `:lo12:` operand to add the
low 12 bits back, `adrp` to a label is only useful when the label is page
aligned.

An `Extern` is a valid target of `b`, `bl`, `adr` and `ldr` literal loads,
which reach +-128MB (`b`, `bl`) or +-1MB (`adr`, `ldr`); `ldr(x0, ext)`
loads from the extern's address. To call a function at any distance, load
its address from the extern's pointer slot, an 8 byte slot in the `externs`
section as on x64: `ldr(x16, a.extern_slot(ext)); blr(x16)`.

Fragments work on aarch64 too. Register holes are `Hole.gp64`,
`Hole.gp32`, `Hole.fp32` and `Hole.fp64`, usable in any register operand,
as a memory base (`mem[src + 8]`, `mem.pre[src - 16]`) and as a plain
64 bit index (`mem[x0 + idx]`). A general purpose hole cannot be filled
with register 31 (`sp`, `xzr`, `wzr`), because whether 31 means the stack
pointer or zero depends on where the register sits. Label holes work as on
x64. Immediate holes are not supported in aarch64 instructions (data
directives still take them), and a hole cannot carry a shift or an extend.

## Compared with DynASM

jita follows DynASM's model (hand-written instructions, labels, sections,
externs, runtime linking) and its instruction tables, and differs in these
ways:

- Everything is known when an instruction is encoded, so there is no
  preprocessor, no action list and no separate `dasm_link`/`dasm_encode`
  step. A `Fragment` gives back the encode-once, fill-in-later behaviour
  where it matters.
- No silent branch relaxation. `jmp`/`jcc` to a label always use rel32,
  `jmp.short` always uses rel8 and linking fails if the target is too far.
- Immediates are range checked by value, and register 31 on aarch64 is
  checked by position, instead of being truncated or reinterpreted.
- Labels are Python objects or strings, and macros are Python functions,
  so local label numbering (`1:`, `<1`, `>1`) is not needed.
- A few instructions are added on x64 (see above) and `.type` struct access
  is `typed()` over `ctypes` definitions.

## Errors

All errors derive from `jita.JitaError`. `EncodeError` is raised when an
instruction cannot be encoded with the given operands (wrong sizes, out of
range immediate, unsupported combination). `LinkError` is raised by `link`
and `load` for unbound labels, out of range branches, missing extern
addresses and misaligned bases. `LoadError` covers executable memory
allocation and writes outside writable sections. Plain `TypeError` is used
for wrong Python types, such as a float where an int operand is expected.
