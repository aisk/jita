# jita reference

jita generates machine code at runtime from ordinary Python calls, the way
DynASM does from a preprocessed C file. This page covers the whole API.
The README has the short version.

Contents: [Assembler](#assembler) · [Sections](#sections) ·
[Labels](#labels) · [Externs](#externs) · [Data](#data-directives) ·
[Functions](#functions) · [Listings](#listings) ·
[x64](#x64) · [Structures](#structures) · [C headers](#c-headers) ·
[aarch64](#aarch64) · [riscv64](#riscv64) · [loongarch64](#loongarch64) ·
[Compared with DynASM](#compared-with-dynasm) ·
[Errors](#errors) · [Type checking](#type-checking)

## Assembler

```python
from jita import Assembler, label
from jita.x64 import *

with Assembler() as a:        # host architecture
    mov(rax, 1)               # module level mnemonics emit into `a`
    ret()
a.mov(rax, 1)                 # every mnemonic is also a method
b = Assembler("x64")          # or "aarch64", "riscv64", "loongarch64", or jita.x64
```

`Assembler(arch)` returns an instance of the architecture's subclass,
`X64Assembler`, `Aarch64Assembler`, `Riscv64Assembler` or
`Loongarch64Assembler` (see [Type checking](#type-checking)).

`with Assembler() as a:`, or `with a:` for an existing one, makes `a` the
current assembler for the module level instruction functions and the
`label()` directive. Contexts nest, so a macro is just a Python function
that emits instructions, and DynASM's `.if` is a Python `if` in the
generator. `jita.current()` returns the active assembler. Outside any
context use the method forms.

Instruction names are lowercase. The x64 mnemonics that clash with Python
keywords or builtins are `and_`, `or_`, `not_` and `int_`; on aarch64
`and_` and `str_` (the methods `a.and_` and `a.str` both exist); on
riscv64 `and_`, `or_`, `not_`, `min_` and `max_` (also `a.min`, `a.max`);
on loongarch64 `and_`, `or_` and `break_`.
`from jita.x64 import *` exports registers and their classes, size
prefixes, `label`, `typed`, the mnemonics and the architecture classes,
and nothing else. `from jita.aarch64 import *` exports registers, their
classes and selectors such as `gp64`, `mem`, the operand types `MemExpr`,
`Mod`, `RegMod` and `Cond`, `label`, `typed`, the mnemonics and the
architecture classes. `from jita.riscv64 import *` exports registers,
their classes, `gpr`, `fpr`, the literal types `Rm`, `CsrName` and
`FenceSet`, `mem`, `MemExpr`, `label`, `typed`, the mnemonics and the
architecture classes, and `from jita.loongarch64 import *` the same
without `mem` and the literal types, with `Fcc` and `Fcsr`. The mnemonic
functions set `__test__ = False`, so pytest does not collect the x64
`test` instruction from a test module that imports it with `*`.

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
size = Extern(ctypes.pythonapi.PyObject_Size)  # name, address from ctypes
cb = Extern("cb", callback)           # a ctypes callback, named explicitly
call(qword[rip + strlen])             # call through the pointer slot
mov(rax, strlen); call(rax)           # 64 bit absolute address
call(strlen)                          # direct rel32, must be within 2GB
fn = a.function(ctypes.c_size_t, ctypes.c_char_p, externs={"strlen": addr})
```

The `externs` mapping given to `link`, `load` or `function` wins over an
address fixed in the constructor. A ctypes function pointer, such as a
function of a `CDLL` or `ctypes.pythonapi` or a `CFUNCTYPE` callback, can
be given instead of an address. `Extern(fn)` takes the name from
`fn.__name__`, and a callback, which has none, is named with
`Extern(name, fn)`. The Extern keeps the function object alive, and so
does a callable from `a.function` or `function` through its assembler. A
`Module` from `a.load()` does not, so keep the function referenced while
the module is in use. `call(ext)` and `jmp(ext)` are rel32 and
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

`a.function(restype, *argtypes, entry=None, externs=None,
functype=ctypes.CFUNCTYPE)` loads the assembler into freshly mapped memory
and returns a `ctypes` function pointer at `entry` (a label or name,
default the start of the image). `externs` supplies extern addresses (see
[Externs](#externs)). The memory is released when the callable is garbage
collected, so closing it is optional (see [Modules](#modules)). Every
`a.function` call loads a separate copy of the code: two callables made
from the same assembler do not share writable data, so for several entries
into one copy load once with `a.load()` and take each entry with
`mod.function`. `functype` is described under
[The GIL and functype](#the-gil-and-functype).

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

`function(restype, *argtypes, entry=None, externs=None, arch=None,
functype=ctypes.CFUNCTYPE)` takes the same arguments as `a.function`, plus
`arch` for the assembler it
creates (default: the host). A body declared with one parameter, `def
f(a)`, receives the assembler for `a.pc`, `a.section`, `a.align` and the
other methods; a body without parameters is called with none. The body's
`__name__`, `__qualname__` and `__doc__` are copied
onto the callable. A decorated function at module level generates its code
when the module is imported.

Either way the callable carries what produced it: `fn.module` is the
loaded `Module` and `fn.assembler` the `Assembler`, for listings, symbol
addresses and writes into data.

### The GIL and functype

`functype` is the ctypes factory that builds the function pointer type,
called as `functype(restype, *argtypes)`. With the default
`ctypes.CFUNCTYPE` the GIL is released while the generated code runs (the
thread state is detached on a free threading build), so the code must not
call the Python C API or read Python objects that another thread may
change. `ctypes.PYFUNCTYPE` keeps the GIL, and if a Python error is set
when the code returns, the call raises it instead of returning the
result. Other factories work too, such as
`functools.partial(ctypes.CFUNCTYPE, use_errno=True)`.

```python
import ctypes
from jita import Extern, function
from jita.x64 import *

@function(ctypes.c_longlong, ctypes.py_object, functype=ctypes.PYFUNCTYPE)
def as_long():                    # tail call into PyLong_AsLongLong
    mov(rax, Extern(ctypes.pythonapi.PyLong_AsLongLong))
    jmp(rax)

print(as_long(42))                # 42
try:
    as_long("x")
except TypeError as e:            # the error PyLong_AsLongLong set
    print(e)
```

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

`mod.function(restype, *argtypes, entry=None, functype=ctypes.CFUNCTYPE)`
returns a callable at `entry` whose `module` attribute keeps the module
alive. `mod.write(where,
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

`jita.aarch64` has `typed` too, with a register, `x0 + 16` or `mem[...]`
as the base.

```python
import ctypes
from jita import Assembler
from jita.aarch64 import *

class Point(ctypes.Structure):
    _fields_ = [("x", ctypes.c_int32), ("y", ctypes.c_int32), ("tag", ctypes.c_uint8),
                ("next", ctypes.c_void_p), ("v", ctypes.c_double * 4)]

p = typed(x0, Point)
with Assembler("aarch64"):
    ldr(w1, p.x)                  # [x0]
    ldrb(w2, p.tag)               # [x0, #8]
    ldr(d0, p.v[1])               # [x0, #32]
    ldr(x3, p.next)               # [x0, #16]
    add(x4, x0, Point.v.offset)
    ldr(d1, typed(x4, ctypes.c_double * 4)[x5])        # [x4, x5, lsl #3]
    ldr(w6, typed(x1, ctypes.c_int32 * 8)[w2.uxtw()])  # [x1, w2, uxtw #2]
```

aarch64 memory operands have no size prefix, so a field's `MemExpr`
carries its size and a load or store of another width raises `EncodeError`
(`ldr(x1, p.x)` with a 4 byte `x`); `ldp` and `stp` check the size of one
register. `mem[p.x]` is the same operand, size included. Offsets are
encoded like any other, scaled when aligned and in range, otherwise as
`ldur` within -256..255, so a field that is both misaligned and outside
-256..255 fails to encode and needs its offset added to a register first.
A register index becomes `[base, Xm, lsl #s]` with the element size as
the shift; an `x` index may also be `.sxtx()`, and a `w` index needs
`.uxtw()` or `.sxtw()`. That addressing mode has no displacement and
scales only by the access size, so the array has to start at the base
(not `p.v[x5]` at offset 24) and the scalar accessed must fill the
element: the element itself, or a field at offset 0 as wide as the element
(`typed(x1, Num * 4)[x2].i` with an 8 byte union `Num`). Other
combinations raise `EncodeError`.

A pair must be two elements of one array. A typed operand knows how many
bytes remain to the end of its array (`MemExpr.extent`), and `ldp` or
`stp` raises `EncodeError` when the second register would go past it:
`ldp(d0, d1, p.v[2])` loads `v[2]` and `v[3]`, while `p.v[3]` cannot
start a pair. A structure field is a single slot, even when a field of the
same size follows it (`p.x` and `p.y`) or its structure is an array
element. A nested array ends with its own row, and a zero length array
is unbounded. Register indexed elements never reach a pair, since `ldp` and
`stp` take no index register. The extent is not part of the address, so
it does not show in `str()` and does not affect `==`. To pair two fields
of a structure, such as the `re` and `im` doubles of a complex number,
use the structure's unsized `addr`: `ldp(d0, d1, c.addr)`.

`jita.riscv64` has `typed` with a register, `a0 + 16` or `mem[...]` as
the base.

```python
import ctypes
from jita import Assembler
from jita.riscv64 import *

class Point(ctypes.Structure):
    _fields_ = [("x", ctypes.c_int32), ("y", ctypes.c_int32), ("tag", ctypes.c_uint8),
                ("next", ctypes.c_void_p), ("v", ctypes.c_double * 4)]

p = typed(a0, Point)
with Assembler("riscv64"):
    lw(a1, p.x)                   # 0(a0)
    lbu(a2, p.tag)                # 8(a0)
    fld(fa0, p.v[1])              # 32(a0)
    ld(a3, p.next)                # 16(a0)
    amoadd.w(zero, a1, p.x)       # (a0)
    sh3add(t0, a5, a0)            # t0 = a0 + a5 * 8
    fld(fa1, typed(t0 + Point.v.offset, ctypes.c_double))   # 24(t0)
```

A field's `MemExpr` carries its size, and every load, store, `lr`, `sc`
and AMO checks it against its access width (`ld(a1, p.x)` with a 4 byte
`x` raises `EncodeError`). `mem[p.x]` is the same operand. An offset
outside -2048..2047 is the usual out of range error, and the atomics
take a field at offset 0 of their base only; for another field, `addi`
its offset into a register and view the field there. `jalr` rejects a
typed operand, since it would jump to the field instead of loading it:
load the field into a register and `jalr` that register. RISC-V loads
and stores have no index register, so an array takes constant indexes
only, and `p.v[a5]` raises `EncodeError` with the code that puts the
element address in a register instead (`sh3add` for 8 byte elements, as
above, or `slli` and `add`, or `mul` for other sizes).

`jita.loongarch64` has `typed` with a register, `a0 + 16` or an unsized
`MemExpr` such as a view's `addr` as the base. LoongArch assembly has no
memory operand, so a field's `MemExpr` stands for the base and offset
operands of the instruction, and an element with a register index for
the base and index operands of `ldx`, `stx`, `fldx` and `fstx`. The same
`MemExpr` can be built directly, `MemExpr(a0, 8)` or
`MemExpr(a0, index=a1)`, and is then an unsized operand equivalent to
writing the pair.

```python
import ctypes
from jita import Assembler
from jita.loongarch64 import *

class Point(ctypes.Structure):
    _fields_ = [("x", ctypes.c_int32), ("y", ctypes.c_int32), ("tag", ctypes.c_uint8),
                ("next", ctypes.c_void_p), ("v", ctypes.c_double * 4)]

p = typed(a0, Point)
with Assembler("loongarch64"):
    ld.w(a1, p.x)                 # ld.w $a1, $a0, 0
    ld.bu(a2, p.tag)              # ld.bu $a2, $a0, 8
    fld.d(fa0, p.v[1])            # fld.d $fa0, $a0, 32
    ldptr.d(a3, p.next)           # ldptr.d $a3, $a0, 16
    amadd_db.w(zero, a1, p.x)     # amadd_db.w $zero, $a1, $a0
    ldx.bu(a4, typed(a5, ctypes.c_uint8 * 64)[a6])  # ldx.bu $a4, $a5, $a6
    alsl.d(t0, a6, a0, 3)         # t0 = a0 + a6 * 8
    fld.d(fa1, typed(t0 + Point.v.offset, ctypes.c_double))  # fld.d $fa1, $t0, 24
```

The loads and stores, `ldptr`, `stptr`, `ll`, `sc`, `preld`, the `am*`
atomics and the bounds checked `ldgt`, `ldle`, `stgt` and `stle` (and
their floating point forms) take a `MemExpr` in place of their address
operands, and the listing shows those operands. The access width comes
from the mnemonic (`ld.w` is 4 bytes, `ammax.du` 8, `fld.s` 4) and must
match the field. Offsets are range checked as for the plain operands,
multiples of 4 up to +-32KB for `ldptr`, `stptr`, `ll` and `sc`. `am*`
and the bounds checked forms take a bare base, so they take a field at
offset 0 only; `addi.d` the offset of another field into a register and
view it there. The indexed instructions add the index without scaling
it and take no offset, so a register index needs 1 byte elements in an
array at the base; other element sizes raise `EncodeError` with the code
that computes the element address (`alsl.d` for 2, 4, 8 and 16 byte
elements, `mul.d` for others). An indexed element needs `ldx`, `stx`,
`fldx`, `fstx` or `preldx` and a field `ld`, `st`, `fld`, `fst` or
`preld`; the error names the right one. A view's unsized `addr` works as
an operand without a width check, and `a0 + 8` is only a base for
`typed`: write `ld.w(a1, a0, 8)` for a plain load.

## C headers

`jita.cheader` reads C headers at runtime and builds the ctypes types
that `typed()` takes, so code for an existing C API uses the header's own
layouts. It needs pycparser and pcpp: `pip install "jita[cheader]"`.

```python
import ctypes, sysconfig
from jita import function
from jita.cheader import load, python_defines
from jita.x64 import *

h = load("Python.h", include_dirs=[sysconfig.get_path("include")], defines=python_defines())

@function(ctypes.c_ssize_t, ctypes.c_void_p)
def py_len():
    mov(rax, typed(rdi, h.PyVarObject).ob_size)   # Py_SIZE(ob)
    ret()

print(py_len(id([1, 2, 3])))                          # 3
print(h.Py_TPFLAGS_HEAPTYPE, h.PyUnicode_1BYTE_KIND)  # 512 1
```

`load(*headers, include_dirs=(), defines=None, types=None, names=None,
exclude=(), strict=False)` preprocesses the headers as one translation
unit and returns a `Header`, a module (not in `sys.modules`) whose
attributes are the C names:

| C | attribute |
| --- | --- |
| `struct foo`, `union foo` | `struct_foo`, `union_foo`: a `ctypes.Structure` or `Union` subclass |
| `typedef ... name` | `name`: the same class, or the ctypes type (`Py_ssize_t` is `c_ssize_t`) |
| `enum foo { A, B }` | `enum_foo`, a `c_int` subclass (wider if the values need it), and `A`, `B` as ints |
| `int f(char *);` | `f`: a `CFUNCTYPE` class, usable with `ctypes.cast(address, h.f)` |
| `#define N (1 << 4)` | `N`: an int, float or str, when the expansion is a constant |

Each header is looked up as given, then in `include_dirs`, which also
serve its own `#include` and `__has_include`. Standard names such as
`int32_t`, `size_t` and `wchar_t` map to their ctypes types; `char *` is
`c_char_p`, a function pointer a `CFUNCTYPE` class, `T[]` is `T * 0`. A
struct that is only forward declared is a class without `_fields_`,
usable through `POINTER`. Anonymous members get `_jita_anonN` names in
`_anonymous_`, so their fields are reached directly, as in C. Bit fields,
`__attribute__((packed))`, `aligned(N)`, `__declspec(align(N))` and
`#pragma pack` (also written `_Pragma` or `__pragma`) become `_fields_`
widths, `_pack_` and `_align_`. An enum field reads back as an instance
of its enum class, the ctypes rule for subclasses of simple types.

The preprocessor defines the host compiler's macros: GCC's (clang's on
macOS) with the sizes ctypes reports, MSVC's on Windows. `<...>`
includes missing from `include_dirs` are skipped, with the usual system
type names declared as types of unknown size; a missing `"..."` include
raises `HeaderError`. `defines={"NAME": value}` adds a macro, a value of
None removes one. `types={"off_t": ctypes.c_int64}` maps a type name
ahead of the headers' typedefs. `names` and `exclude` are fnmatch
patterns choosing the names exposed; the types they use are built anyway.
Every call parses again and creates new classes, so keep the `Header` to
share them. For Python.h, `python_defines()` gives the running
interpreter's layout flags (`Py_GIL_DISABLED`, `Py_DEBUG`,
`Py_TRACE_REFS`, `Py_STATS`, each only when set) as `defines`, since
pyconfig.h does not always record them: on Windows one pyconfig.h serves
the GIL and free threaded builds.

What cannot be converted exactly is skipped, never loaded with a wrong
layout: variadic functions, `__int128`, `typeof`, zero width bit fields,
bit fields in packed structs (GCC and MSVC pack them differently),
packed enums, `#pragma pack` inside a struct body, a field aligned beyond
its natural alignment, `mode`, `vector_size` and other attributes that
change a layout, system types used by value unless named in `types`,
declarations pycparser cannot parse (C23, `_Generic`), and everything
that depends on a skipped type. `diagnostics(h)` lists
`Diagnostic(severity, kind, name, file, line, message)` records:
`"skip"` for a declaration left out, `"note"` for something ignored or
degraded, such as a skipped system include, a pointer to an unsupported
type made `c_void_p` or a macro hidden by a declaration of the same name.
Reading a skipped name raises an `AttributeError` with the reason and the
location, and `strict=True` raises `HeaderError` at the first skip.
Function like macros and macros using `sizeof` are not exposed.

Layouts are the ones ctypes computes on the host, so there is no target
option. The 64 bit Linux targets (x86-64, aarch64, riscv64 and
loongarch64) share one layout, so offsets loaded on one are right for the
others' code; Windows and macOS differ in `long`, `wchar_t` and
`long double`.

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
`ldr w0`, `ldr d0`), so memory operands have no size prefix. The operands
of `typed()` fields carry their size and are checked against the access
(see Structures).

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

## riscv64

`jita.riscv64` covers RV64 with the I, M, A, F, D, Zicsr, Zifencei, Zba
and Zbb extensions. `Assembler()` picks it on a riscv64 host;
`Assembler("riscv64")` generates riscv64 code anywhere.

```python
from jita import Assembler
from jita.riscv64 import *

a = Assembler("riscv64")
with a:                           # int64_t sum(int64_t *p, size_t n)
    mv(a2, zero)
    beqz(a1, "done")
    label("loop")
    ld(a3, mem[a0])
    add(a2, a2, a3)
    addi(a0, a0, 8)
    addi(a1, a1, -1)
    bnez(a1, "loop")
    label("done")
    mv(a0, a2)
    ret()
```

Registers have their ABI names, `zero ra sp gp tp t0`..`t6 s0`..`s11
a0`..`a7` and `ft0`..`ft11 fs0`..`fs11 fa0`..`fa7`; `x0`..`x31` and
`f0`..`f31` are the same objects under their numbers, and `fp` is `s0`.
There are two classes, `X` and `F`: the mnemonic decides the width
(`lw`, `ld`, `fadd.s`, `fadd.d`). `gpr(n)` and `fpr(n)` select a register
by number. Listings print the ABI names, as objdump does.

| Operand | jita | GNU as |
| --- | --- | --- |
| immediate | `addi(a0, a1, -1)`, `slli(a0, a0, 3)`, `lui(a0, 0x12345)` | `-1`, `3`, `0x12345` |
| base, offset | `mem[a0]`, `mem[sp + 8]`, `mem[sp - 16]` | `0(a0)`, `8(sp)`, `-16(sp)` |
| atomic address | `lr.w(a0, mem[a1])` | `(a1)` |
| dotted mnemonic | `fadd.d(fa0, fa1, fa2)`, `lr.w.aq(...)`, `fence.i()` | `fadd.d`, `lr.w.aq`, `fence.i` |
| rounding mode | `fcvt.w.d(a0, fa0, rm="rtz")` | `fcvt.w.d a0, fa0, rtz` |
| CSR | `csrr(a0, "fflags")`, `csrrw(a0, 0x7c0, a1)` | `fflags`, `0x7c0` |
| fence sets | `fence("rw", "w")`, `fence()` for `iorw, iorw` | `fence rw, w` |
| label | `beqz(a0, "loop")`, `j(lbl)`, `call("f")` | `beqz a0, loop` |
| keyword mnemonics | `and_`, `or_`, `not_`, `min_`, `max_` (also `a.min(...)`) | `and`, `or`, `not`, `min`, `max` |

Mnemonics with dots are attributes of a namespace object: `fadd` itself
is not an instruction, `fadd.s` and `fadd.d` are. `fence`, `fmv.d`,
`lr.w`, `sc.d` and the AMOs are both, so `fence()`, `fence.i()` and
`amoadd.w.aqrl(...)` all work, as do the method forms `a.fadd.d(...)`.
Immediates are checked by value against their field: 12 bit signed for
I and S-type, 0..63 for shift amounts (0..31 for the `*w` forms), 5 bit
unsigned for `csrrwi` and friends, 12 bit CSR numbers. `lui` and `auipc`
take the 20 bit field value like GNU as, `0..0xfffff`, and also its
negative two's complement spelling (`lui(a0, -1)` is `lui a0, 0xfffff`),
which GNU as rejects. Floating point instructions that round take `rm=`
(`rne rtz rdn rup rmm dyn`, typed as `Rm`); without it the mode is `dyn`,
except for the exact conversions `fcvt.d.w`, `fcvt.d.wu` and `fcvt.d.s`,
which take no rounding mode, as in GNU as. CSR names are `fflags frm fcsr
cycle time instret` (`CsrName`); other CSRs are given by number.

Beyond the base instructions jita has the GNU as pseudo instructions
`nop mv not_ neg negw sext.w zext.b zext.w seqz snez sltz sgtz sgt
sgtu`, the branches `beqz bnez blez bgez bltz bgtz bgt ble bgtu bleu`,
`j`, `jr(rs)` and `jr(rs, offset)`, `ret`, `jal(lbl)`, `jalr(rs)` and
`jalr(rs, offset)` with `ra` as the link register, `fmv fneg fabs` in
both precisions, `fgt` and `fge` (`flt`/`fle` with the operands
swapped), `fmv.x.s` and `fmv.s.x` (the old names of `fmv.x.w` and
`fmv.w.x`), `csrr csrw csrs csrc csrwi csrsi csrci`, `frcsr fscsr frrm
fsrm fsrmi frflags fsflags fsflagsi`, `rdcycle rdtime rdinstret`,
`fence.tso` and `unimp`, and these multi instruction ones:

- `li(rd, value)` loads any 64 bit value, signed or unsigned, in 1 to 8
  instructions. The expansion is GNU as' (`lui`, `addiw`, `slli`,
  `addi`), so listings and disassembly agree with assembler output.
- `lla(rd, target)` is `auipc rd; addi rd, rd, lo`. `la` is the same:
  jita has no GOT, as GNU as without PIC.
- `call(target)` is `auipc ra; jalr ra, lo(ra)` and `tail(target)` is
  `auipc t1; jalr zero, lo(t1)`.
- A load with a label instead of a memory operand, `ld(rd, target)`, is
  `auipc rd; ld rd, lo(rd)`. Stores and floating point loads name the
  register for the `auipc`: `sd(a0, target, t0)`, `fld(fa0, target, t0)`.
  That register cannot be `zero`, which would drop the high part (only
  `lla(zero, ...)` is accepted, as in GNU as), and for an integer store
  it cannot be the register stored, which the `auipc` would overwrite
  (`sd(t0, target, t0)`; GNU as assembles it anyway).

The listing shows a pseudo instruction as one line with all its words.

Branches (`beq`..`bgeu` and their pseudo forms) reach +-4KB, `jal` and
`j` +-1MB, and the `auipc` pairs about +-2GB (the high part is rounded so
that the sign extended low 12 bits bring it back). Branches are never
relaxed; out of range is a `LinkError`. An `Extern` is a valid target of
all of them, like a label, so `call(ext)` works for a function within
+-2GB and `ld(a0, ext)` loads the 8 bytes at the extern's address, as
`ldr(x0, ext)` does on aarch64. To call a function at any distance, load
its address from the extern's pointer slot: `ld(t0, a.extern_slot(ext));
jalr(t0)`.

`typed(a0, T)` views memory as a ctypes structure, and its fields are
`MemExpr`s that carry their size, checked against the access width (see
Structures).

Compressed instructions, the vector extension, `%hi`/`%lo` operators and
GOT or PLT references are not supported. On a riscv64 host, loading code
flushes the instruction cache through libgcc's `__clear_cache`, or the
`riscv_flush_icache` system call when libgcc is not found.

## loongarch64

`jita.loongarch64` covers LA64: the base integer instructions, the
atomics (`ll`, `sc`, `am*` and `am*_db`, barriers) and scalar single and
double floating point. `Assembler()` picks it on a loongarch64 host;
`Assembler("loongarch64")` generates loongarch64 code anywhere.

```python
from jita import Assembler
from jita.loongarch64 import *

a = Assembler("loongarch64")
with a:                           # int64_t sum(int64_t *p, size_t n)
    move(a2, zero)
    beqz(a1, "done")
    label("loop")
    ld.d(a3, a0, 0)
    add.d(a2, a2, a3)
    addi.d(a0, a0, 8)
    addi.d(a1, a1, -1)
    bnez(a1, "loop")
    label("done")
    move(a0, a2)
    ret()
```

Registers have their ABI names, `zero ra tp sp a0`..`a7 t0`..`t8 fp
s0`..`s8` and `fa0`..`fa7 ft0`..`ft15 fs0`..`fs7`; `r0`..`r31` and
`f0`..`f31` are the same objects under their numbers, `s9` is `fp`, and
r21, which has no ABI name, is `r21`. The condition flags are
`fcc0`..`fcc7` and the FP control registers `fcsr0`..`fcsr3`. The
classes are `R`, `F`, `Fcc` and `Fcsr`: the mnemonic decides the width
(`ld.w`, `ld.d`, `fadd.s`, `fadd.d`). `gpr(n)` and `fpr(n)` select a
register by number. Python names have no `$`; listings print registers
as objdump does, `$a0`.

| Operand | jita | GNU as |
| --- | --- | --- |
| register | `add.d(a0, a1, a2)` | `add.d $a0, $a1, $a2` |
| immediate | `addi.d(a0, a1, -1)`, `slli.d(a0, a0, 3)`, `lu12i.w(a0, -1)` | `-1`, `3`, `-1` |
| base, offset | `ld.d(a0, sp, 8)`, `st.w(a1, a0, 0)`, `ldptr.d(a0, a1, 4096)` | `ld.d $a0, $sp, 8` |
| base, index | `ldx.d(a0, a1, a2)` | `ldx.d $a0, $a1, $a2` |
| dotted mnemonic | `fcmp.clt.d(fcc0, fa0, fa1)`, `amadd_db.w(...)`, `crc.w.b.w(...)` | `fcmp.clt.d`, `amadd_db.w` |
| digit part | `revb._2h(a0, a1)`, `bitrev._8b(a0, a1)` | `revb.2h`, `bitrev.8b` |
| label | `beqz(a0, "loop")`, `b(lbl)`, `call36("f")` | `beqz $a0, loop` |
| keyword mnemonics | `and_`, `or_`, `break_` | `and`, `or`, `break` |

There is no `mem[...]`: as in GNU syntax a load or store takes the base
register and the offset as plain operands. A `MemExpr(base, offset)`, or
`MemExpr(base, index=reg)` for `ldx` and the other indexed forms, stands
for that pair and encodes the same, `ld.w(a0, MemExpr(a1, 8))` is
`ld.w(a0, a1, 8)`. The fields of `typed(a0, T)` are such `MemExpr`s that
also carry their size (see Structures). Mnemonics with dots are
attributes of a namespace object, so `add` itself is not an instruction,
`add.w` and `add.d` are, and the methods are `a.add.d(...)`. A dotted
part that starts with a digit gets a leading underscore. Immediates are
checked by value against their field as GNU as checks them: signed for
`addi`, `slti`, `sltui`, `lu52i.d`, `addu16i.d`, loads and stores and
the 20 bit `lu12i.w`, `lu32i.d` and `pcadd*` (`lu12i.w(a0, -1)`, not
`0xfffff`); unsigned for `andi`, `ori`, `xori`, shift amounts, `break`,
`dbar` and the other codes. The offsets of `ll`, `sc`, `ldptr`, `stptr`
and `jirl` are byte offsets and must be multiples of 4. When `ld.w`,
`ld.d`, `st.w` or `st.d` cannot encode an offset that its `ldptr` or
`stptr` form can, or the other way round, the error names that
instruction. `alsl` takes the shift 1..4, `bstrins` and `bstrpick` need
msb >= lsb, and the `am*` instructions (except `amswap.w`) reject an rd
equal to rk or rj, as GNU as does.

Beyond the base instructions jita has the GNU as aliases `nop move ret
jr ud`, `rdcntvl.w rdcntvh.w rdcntid.w`, the branches `bgt ble bgtu bleu`
(operands swapped) and `bltz bgez bgtz blez`, the swapped compares
`fcmp.sgt fcmp.sge fcmp.cugt fcmp.cuge` in both precisions, and these
multi instruction ones:

- `li.w(rd, value)` loads a 32 bit value (signed or unsigned, sign
  extended to 64 bits), `li.d(rd, value)` any 64 bit value, in 1 to 4
  instructions. The expansion is GNU as' (`lu12i.w`, `ori` or `addi.w`,
  `lu32i.d`, `lu52i.d`, each left out when the sign extension of the
  parts below already gives it), so listings and disassembly agree with
  assembler output. jita accepts -2^31..2^32-1 for `li.w` and
  -2^63..2^64-1 for `li.d` and rejects anything else. GNU as is looser:
  a `li.w` value whose high 32 bits are all ones is cut to its low 32
  bits (`li.w $a0, -0x80000001` loads 0x7fffffff, `0xffffffff80000000` is
  accepted), other values beyond 32 bits stop it with "li overflow", and
  `li.d` values beyond 64 bits wrap (`-0x8000000000000001` loads
  0x7fffffffffffffff).
- `la.local(rd, target)` and `la.pcrel(rd, target)` are `pcalau12i rd;
  addi.d rd, rd, lo12`. GNU as' plain `la` goes through the GOT, and jita
  has no GOT, so `la` is only the namespace of these two.
- `call36(target)` is `pcaddu18i ra; jirl ra, ra, lo` and
  `tail36(rj, target)` is `pcaddu18i rj; jirl zero, rj, lo`. `call` and
  `tail` are the same, as in GNU as.
- A load with a label instead of the base and offset, `ld.d(rd, target)`,
  is `pcalau12i rd; ld.d rd, rd, lo12`. Stores and floating point loads
  and stores name the register for the `pcalau12i`:
  `st.d(a0, target, t0)`, `fld.d(fa0, target, t0)`. For an integer
  store that register cannot be the one stored (`st.d(t0, target, t0)`),
  which the `pcalau12i` would overwrite. GNU as has no macro for these;
  they are the pair GCC emits.
- `pcaddi(rd, target)` is the single instruction form, +-2MB.

The register that holds the address of a label reference cannot be
`zero`: the high part would be lost and the second instruction would use
an absolute address. GNU as accepts `tail36 $zero, lbl`; jita rejects it.
`la.local(zero, ...)` leaves no register behind and is accepted.

`pcalau12i` works on 4KB pages like `adrp` on aarch64; the pair adds the
low 12 bits of the target, and the page part is rounded so that their
sign extension brings it back. The pair reaches about +-2GB and
`call36`/`tail36` about +-128GB, both computed from the exact field
ranges. `beq`..`bgeu` and their aliases reach +-128KB, `beqz`, `bnez`,
`bceqz` and `bcnez` +-4MB, and `b` and `bl` +-128MB. Branches are never
relaxed; out of range is a `LinkError`. An `Extern` is a valid target of
all of them, like a label, so `call36(ext)` works within +-128GB and
`ld.d(a0, ext)` loads the 8 bytes at the extern's address. To call a
function at any distance, load its address from the extern's pointer
slot: `ld.d(t0, a.extern_slot(ext)); jirl(ra, t0, 0)`.

The vector extensions (LSX, LASX), binary translation (LBT), privileged
instructions, the newer `amcas`, `am*.b`/`am*.h`, `sc.q`, `llacq`/`screl`,
`frecipe` and `frsqrte`, and the GOT, TLS and large code model forms of
`la` are not supported. On a loongarch64 host, loading code runs libgcc's
`__clear_cache`, which is an `ibar 0`, or an `ibar 0` stub of jita's own
when libgcc is not found.

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
- Upstream DynASM has no RISC-V module. The riscv64 table is written from
  the specification and checked against GNU as, and pseudo instructions
  that expand to several instructions (`li`, `lla`, `call`, `tail`, label
  loads and stores) are part of the instruction set instead of being left
  to macros. The same goes for loongarch64 (`li.d`, `la.local`, `call36`),
  whose table is checked against GNU as, not taken from Loongson's
  DynASM port.

## Errors

All errors derive from `jita.JitaError`. `EncodeError` is raised when an
instruction cannot be encoded with the given operands (wrong sizes, out of
range immediate, unsupported combination). `LinkError` is raised by
`link`, `load` and `function` for unbound labels, out of range branches,
missing extern addresses and misaligned bases. `LoadError` covers
executable memory allocation and writes outside writable sections. Plain
`TypeError` is used for wrong Python types, such as a float where an int
operand is expected. `jita.cheader.HeaderError` is raised for a header
that cannot be found or loaded.

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
classes (see the operand tables of [x64](#x64), [aarch64](#aarch64) and
[riscv64](#riscv64) and [loongarch64](#loongarch64)), and each mnemonic
has one overload per combination
of classes the encoder accepts. So a checker reports wrong operand
classes and widths, a wrong number of operands, a register where memory
is needed and the reverse, misspelled mnemonics, `.short` on a
non-branch, misspelled `b.cond` attributes and condition codes
(`csel(x0, x1, x2, "lx")`), on riscv64 misspelled dotted mnemonics,
rounding modes, CSR names and fence sets, and on loongarch64 misspelled
dotted mnemonics and fcmp conditions and a register of the wrong class
(`fcsr0` where `fcc0` is expected).

What depends on values stays a runtime `EncodeError`: immediate ranges and
encodability (imm8 versus imm32, bitmask and FP immediates), instructions
that need one specific register (`cl` as a shift count, `xmm0` for
`blendvps`, accumulator forms of `mov64`), `sp` versus `xzr` on aarch64,
the shape of memory operands (a rip base with an index, offset ranges,
writeback into a transferred register) and whether a label is ever bound.
`typed()` fields are typed `Any`, as are the names of a `jita.cheader`
`Header`, which exist only once the header is loaded. The functions
returned by `function` and `a.function` accept any arguments
(`JitFunction` from `jita.runtime`), since ctypes decides the signature
at runtime. A `MemExpr(...)` built by hand has no size class and is
rejected where a sized operand is expected; build memory operands with
`qword[...]` and friends.

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

`Assembler("x64")`, `Assembler("aarch64")`, `Assembler("riscv64")` and
`Assembler("loongarch64")` are typed as `X64Assembler`,
`Aarch64Assembler`, `Riscv64Assembler` and `Loongarch64Assembler`, whose
methods have the same overloads as the module level mnemonics; the
riscv64 and loongarch64 namespaces (`fadd`, `fcvt.w`, `add`, `fcmp.clt`)
are protocols with a method per mnemonic, and `rm=` is typed as `Rm`. `Assembler()` (host) and
`Assembler(jita.x64)` are a plain `Assembler`, on which any method name
and operands are accepted; annotate the parameter of a `function` body as
`X64Assembler` to get the checks there. On the typed assemblers a misspelled method is reported
as not callable (`a.movv(rax, 1)`).

The stubs `jita/x64/insns.pyi`, `jita/aarch64/insns.pyi`,
`jita/riscv64/insns.pyi`, `jita/loongarch64/insns.pyi` and the package
`__init__.pyi` files are
generated by `tools/gen_stubs.py`, which asks the encoder which operand
classes each mnemonic accepts. Run it after changing
the instruction tables; `--check` only reports stale stubs.
