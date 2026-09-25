# jita reference

jita generates machine code at runtime from ordinary Python calls, the way
DynASM does from a preprocessed C file. This page covers the whole API.
The README has the short version.

Contents: [Assembler](#assembler) · [Sections](#sections) ·
[Labels](#labels) · [Externs](#externs) · [Data](#data-directives) ·
[Functions](#functions) · [Listings](#listings) ·
[x64](#x64) · [Structures](#structures) ·
[aarch64](#aarch64) · [Compared with DynASM](#compared-with-dynasm) ·
[Errors](#errors) · [Type checking](#type-checking)

## Assembler

```python
from jita import Assembler, label
from jita.x64 import *

with Assembler() as a:        # host architecture
    mov(rax, 1)               # module level mnemonics emit into `a`
    ret()
a.mov(rax, 1)                 # every mnemonic is also a method
b = Assembler("x64")          # or "aarch64", or the package jita.x64
```

`Assembler(arch)` returns an instance of the architecture's subclass,
`X64Assembler` or `Aarch64Assembler` (see [Type checking](#type-checking)).

`with Assembler() as a:`, or `with a:` for an existing one, makes `a` the
current assembler for the module level instruction functions and the
`label()` directive. Contexts nest, so a macro is just a Python function
that emits instructions, and DynASM's `.if` is a Python `if` in the
generator. `jita.current()` returns the active assembler. Outside any
context use the method forms.

Instruction names are lowercase. The x64 mnemonics that clash with Python
keywords or builtins are `and_`, `or_`, `not_` and `int_`; on aarch64
`and_` and `str_` (the methods `a.and_` and `a.str` both exist).
`from jita.x64 import *` exports registers and their classes, size
prefixes, `label`, `typed`, the mnemonics and the architecture classes,
and nothing else. Note that it exports `test`,
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
(`img.address("entry")`, `a.function(..., entry="entry")`). A label used as
a `mov r64` immediate or as a `qword` value becomes its 64 bit absolute
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
fn = a.function(ctypes.c_size_t, ctypes.c_char_p, externs={"strlen": addr})
```

The `externs` mapping given to `link`, `load` or `function` wins over an
address fixed in the constructor. `call(ext)` and `jmp(ext)` are rel32 and
fail with `LinkError` when the target is more than 2GB from the code,
which is common for shared libraries. An Extern used as a rip-relative
memory operand refers instead to an 8 byte slot holding the extern's
address. jita creates one slot per extern name in a read-only `externs`
section laid out next to the code (`a.extern_slot(ext)` returns its
label), so `call(qword[rip + ext])`, `jmp(qword[rip + ext])`, `mov(rax,
qword[rip + ext])` and `lea(rax, ptr[rip + ext])` work at any distance. A
displacement on a slot operand is an error.

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

## Functions

The usual path from generated code to something Python can call is
`a.function`:

```python
import ctypes
from jita import Assembler
from jita.x64 import *

with Assembler() as a:
    lea(rax, ptr[rdi + rsi])      # int64_t add(int64_t x, int64_t y)
    ret()
add2 = a.function(ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
print(add2(40, 2))                # 42
```

`a.function(restype, *argtypes, entry=None, externs=None)` loads the
assembler into freshly mapped memory and returns a `ctypes` function
pointer at `entry` (a label or name, default the start of the image).
`externs` supplies extern addresses (see [Externs](#externs)). The memory
is released when the callable is garbage collected, so closing it is
optional (see [Modules](#modules)). Every `a.function` call loads a
separate copy of the code: two callables made from the same assembler do
not share writable data, so for several entries into one copy load once
with `a.load()` and take each entry with `mod.function`.

The `function` decorator is one more layer on top. The decorated body runs
once, at decoration time, inside a fresh `Assembler`, and the decorated
name becomes the callable. Inside a factory
the body closes over the factory's parameters, which is how code is
specialized at runtime: values become immediates and Python `if`
statements pick what is emitted.

```python
import ctypes
from jita import function
from jita.x64 import *

def make_scale(k, bias=0):
    @function(ctypes.c_int64, ctypes.c_int64)
    def scale():                  # int64_t scale(int64_t x): x * k + bias
        imul(rax, rdi, k)         # k is an immediate in the code
        if bias:                  # decided while generating, not at runtime
            add(rax, bias)
        ret()
    return scale

triple = make_scale(3)
print(triple(7), make_scale(10, 1)(7))   # 21 71
```

`function(restype, *argtypes, entry=None, externs=None, arch=None)` takes
the same arguments as `a.function`, plus `arch` for the assembler it
creates (default: the host). A body declared with one parameter, `def
f(a)`, receives the assembler for `a.pc`, `a.section`, `a.align` and the
other methods; a body without parameters is called with none. The body's
`__name__`, `__qualname__` and `__doc__` are copied
onto the callable. A decorated function at module level generates its code
when the module is imported.

Either way the callable carries what produced it: `fn.module` is the
loaded `Module` and `fn.assembler` the `Assembler`, for listings, symbol
addresses and writes into data.

### Modules

`a.load(externs=None)` is the step under `a.function`. It links the
assembler at the address of freshly mapped memory, copies the image in,
marks the read-only prefix executable and returns a `Module`. Use it when
one piece of code has several entry points or data that Python updates.

```python
import ctypes
from jita import Assembler
from jita.x64 import *

with Assembler() as a:
    label("get")
    mov(rax, qword[rip + "counter"])
    ret()
    label("bump")
    add(qword[rip + "counter"], 1)
    ret()
    with a.section("vars", writable=True):
        a.align(8)
        label("counter")
        a.qword(0)

mod = a.load()
get = mod.function(ctypes.c_int64, entry="get")
bump = mod.function(None, entry="bump")
mod.write("counter", (40).to_bytes(8, "little"))
bump(); bump()
print(get(), hex(mod.address("counter")))   # 42 and the address
```

`mod.function(restype, *argtypes, entry=None)` returns a callable at
`entry` whose `module` attribute keeps the module alive. `mod.write(where,
data)` overwrites bytes at a label, a symbol name or an offset from the
image base; only writable sections accept writes, the executable prefix is
a `LoadError`. `mod.address(label)` returns an absolute address and
`mod.image` is the linked `Image`. The memory is unmapped when the module is
garbage collected, or earlier by `mod.close()` or by leaving a
`with a.load() as mod:` block. Calling a function of a closed module crashes
the process.

A raw address taken from a function, such as
`ctypes.cast(fn, ctypes.c_void_p).value`, a pointer stored in C, or an
extern passed to another module, does not keep the module alive, so keep
the callable or the module referenced as long as the address is in use.

### Linking without loading

```py
img = a.link(base=0x1000, externs={...})   # Image: bytes plus addresses
img.data, img.address("entry"), img.section_offsets, img.symbols
```

`link` resolves every label and extern patch for the given base and
returns an `Image` without mapping anything, for example to write the bytes
elsewhere or to print a listing at chosen addresses. An assembler can be
linked or loaded more than once, and labels bound after linking are
detected.

## Listings

```py
from jita.tools.listing import listing
print(listing(a))             # offsets, bytes, one instruction per line
print(listing(a, img))        # with linked addresses
print(listing(fn.assembler, fn.module.image))   # a function at its real addresses
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
| register classes | `Gp8 Gp16 Gp32 Gp64 Xmm Ymm St Rip`, `Gp` for any of the four gp widths | |
| by number | `gp8(n) gp16(n) gp32(n) gp64(n) xmm(n) ymm(n) st(n)` | `Rb(n) Rw(n) Rd(n) Rq(n)` |
| memory | `qword[rbx + rcx*8 + 8]`, `dword[rax]`, `byte[rip + lbl]` | `qword [rbx+rcx*8+8]` |
| size from the other operand | `ptr[rbx]` as in `mov(rax, ptr[rbx])` | `[rbx]` |
| sizes | `byte word dword qword oword yword tword ptr` | `byte word dword qword oword yword` |
| memory classes | `Mem8 Mem16 Mem32 Mem64 Mem80 Mem128 Mem256`, `MemAny` for `ptr[...]` and `rbx + 8`, `Mem` for any | |
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
`lr` and `fp` are `x30` and `x29`. Their classes are `X` (with `xzr`), `W`
(with `wzr`), `Sp`, `S`, `D` and `Q`, with `Gp` for `X | W` and `Fp` for
the FP ones; a shifted or extended register is a `RegMod[X]` or
`RegMod[W]`. `gp64(n)`, `gp32(n)`, `fp32(n)`,
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

Condition codes are `eq ne cs hs cc lo mi pl vs vc hi ls ge lt gt le al`,
typed as `Cond`.
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

## Compared with DynASM

jita follows DynASM's model (hand-written instructions, labels, sections,
externs, runtime linking) and its instruction tables, and differs in these
ways:

- Everything is known when an instruction is encoded, so there is no
  preprocessor, no action list and no separate `dasm_link`/`dasm_encode` step.
  Since the encoder itself runs at runtime, DynASM's encode-once templates
  for runtime values (`Rq(n)`, runtime `imm`) are not needed either;
  `gp64(n)` and plain Python values do that job.
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
range immediate, unsupported combination). `LinkError` is raised by
`link`, `load` and `function` for unbound labels, out of range branches,
missing extern addresses and misaligned bases. `LoadError` covers
executable memory allocation and writes outside writable sections. Plain
`TypeError` is used for wrong Python types, such as a float where an int
operand is expected.

## Type checking

jita ships `py.typed` and type stubs, so pyright, mypy and editors see
every register, size prefix and mnemonic with its accepted operands.

```py
from jita import Assembler
from jita.x64 import *

add(rax, rcx)                 # fine
add(rax, ecx)                 # error: no overload of add for Gp64, Gp32
movzx(eax, ptr[rdi])          # error: movzx needs byte[...] or word[...]
a = Assembler("x64")          # an X64Assembler
a.mov(eax, qword[rdi])        # error: mixed operand sizes
```

Registers are instances of width classes and memory operands of sized
classes (see the operand tables of [x64](#x64) and [aarch64](#aarch64)),
and each mnemonic has one overload per combination of classes the encoder
accepts. So a checker reports wrong operand classes and widths, a wrong
number of operands, a register where memory is needed and the reverse,
misspelled mnemonics, `.short` on a non-branch, misspelled `b.cond`
attributes and condition codes (`csel(x0, x1, x2, "lx")`).

What depends on values stays a runtime `EncodeError`: immediate ranges and
encodability (imm8 versus imm32, bitmask and FP immediates), instructions
that need one specific register (`cl` as a shift count, `xmm0` for
`blendvps`, accumulator forms of `mov64`), `sp` versus `xzr` on aarch64,
the shape of memory operands (a rip base with an index, offset ranges,
writeback into a transferred register) and whether a label is ever bound.
`typed()` fields are typed `Any`, and the functions returned by
`function` and `a.function` accept any arguments (`JitFunction` from
`jita.runtime`), since ctypes decides the signature at runtime. A
`MemExpr(...)` built by hand has no size class and is rejected where a
sized operand is expected; build memory operands with `qword[...]` and
friends.

Annotate helpers with the concrete classes, or leave the parameters
unannotated:

```python
from jita.x64 import *
from jita.x64 import Gp64, Mem64, MemAny, X64Assembler

def load(dst: Gp64, src: Mem64 | MemAny) -> None:
    mov(dst, src)

def epilogue(a: X64Assembler) -> None:
    a.ret()
```

mypy skips the bodies of functions without annotations, so with mypy
give helpers and `function` bodies a return annotation (`-> None`) or set
`check_untyped_defs = true`; pyright checks them either way.

`Reg` and `MemExpr` are too wide for an operand parameter: `mov(dst, src)`
with `dst: Reg` is an error, because some registers do not fit. A union
such as `Gp` works where every member does (`inc(r)`), but not for two
operands that must agree in width (`add(r, r)` with `r: Gp` is an error),
since checkers try each member on its own.

`Assembler("x64")` and `Assembler("aarch64")` are typed as `X64Assembler`
and `Aarch64Assembler`, whose methods have the same overloads as the
module level mnemonics. `Assembler()` (host) and `Assembler(jita.x64)` are
a plain `Assembler`, on which any method name and operands are accepted;
annotate the parameter of a `function` body as `X64Assembler` to get the
checks there. A misspelled method is not caught even on `X64Assembler`.

The stubs `jita/x64/insns.pyi`, `jita/aarch64/insns.pyi` and the package
`__init__.pyi` files are generated by `tools/gen_stubs.py`, which asks the
encoder which operand classes each mnemonic accepts. Run it after changing
the instruction tables; `--check` only reports stale stubs.
