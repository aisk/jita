"""typed() on riscv64: ctypes structures as views producing sized memory
operands."""

import ctypes
import platform
import types

import pytest
from oracle import assemble, requires_riscv64_oracle

import jita.riscv64 as R
from jita import Assembler, EncodeError
from jita.riscv64 import *  # noqa: F403
from jita.riscv64.insns import MNEMONICS
from jita.riscv64.structs import Typed, TypedArray
from jita.riscv64.table import MAP_OP

needs_riscv64_host = pytest.mark.skipif(
    platform.machine().lower() != "riscv64" or platform.system() == "Windows",
    reason="needs a riscv64 POSIX host",
)


# c_int32 is c_long on Windows.
I32 = ctypes.c_int32.__name__


class Point(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_int32),
        ("y", ctypes.c_int32),
        ("tag", ctypes.c_uint8),
        ("next", ctypes.c_void_p),
        ("v", ctypes.c_double * 4),
    ]


class Packed(ctypes.Structure):
    _layout_ = "ms"
    _pack_ = 1
    _fields_ = [("a", ctypes.c_uint8), ("b", ctypes.c_int64), ("c", ctypes.c_int16)]


class Inner(ctypes.Structure):
    _fields_ = [("lo", ctypes.c_int16), ("hi", ctypes.c_int64)]


class Num(ctypes.Union):
    _fields_ = [("i", ctypes.c_int64), ("f", ctypes.c_double), ("b", ctypes.c_uint8 * 8)]


class Outer(ctypes.Structure):
    _anonymous_ = ("anon",)
    _fields_ = [
        ("flag", ctypes.c_bool),
        ("inner", Inner),
        ("num", Num),
        ("anon", Inner),
        ("pts", Point * 3),
        ("grid", (ctypes.c_int32 * 3) * 2),
        ("name", ctypes.c_char_p),
        ("ptr", ctypes.POINTER(Point)),
    ]


class Derived(Point):
    _fields_ = [("extra", ctypes.c_uint16)]


class Bits(ctypes.Structure):
    _fields_ = [("n", ctypes.c_int32), ("f", ctypes.c_uint32, 3)]


def M(base, disp=0, size=None):
    return MemExpr(base, disp, size)


def enc(fn, *ops):
    a = Assembler(R)
    fn(*ops, asm=a)
    return bytes(a.cur.buf)


def test_spec_example():
    p = typed(a0, Point)
    assert p.x == M(a0, 0, 4)
    assert p.y == M(a0, 4, 4)
    assert p.tag == M(a0, 8, 1)
    assert p.next == M(a0, 16, 8)
    assert p.v[1] == M(a0, 32, 8)
    assert p.v.addr == M(a0, 24)
    assert p.v.addr.size is None
    assert p.size == ctypes.sizeof(Point) and p.ctype is Point
    assert p.addr == mem[a0]
    assert str(p.v[1]) == "32(a0)"
    # The size is part of the operand, not of its text.
    assert p.x != mem[a0] and str(p.x) == str(mem[a0])


@pytest.mark.parametrize(
    "ctype,size",
    [
        (ctypes.c_int8, 1), (ctypes.c_uint8, 1), (ctypes.c_char, 1), (ctypes.c_bool, 1),
        (ctypes.c_int16, 2), (ctypes.c_uint16, 2),
        (ctypes.c_int32, 4), (ctypes.c_uint32, 4), (ctypes.c_float, 4),
        (ctypes.c_int64, 8), (ctypes.c_uint64, 8), (ctypes.c_double, 8),
        (ctypes.c_void_p, 8), (ctypes.c_char_p, 8), (ctypes.c_size_t, 8),
        (ctypes.c_ssize_t, 8), (ctypes.POINTER(ctypes.c_int), 8),
        (ctypes.CFUNCTYPE(None), 8),
    ],
)  # fmt: skip
def test_scalar_sizes(ctype, size):
    class S(ctypes.Structure):
        _fields_ = [("pad", ctypes.c_int64), ("f", ctype)]

    assert typed(s3, S).f == M(s3, 8, size)
    assert typed(s3, ctype) == M(s3, 0, size)


def _paths(ctype, prefix=()):
    """Every scalar leaf of a ctypes type as (path, offset, size)."""
    if issubclass(ctype, (ctypes.Structure, ctypes.Union)):
        for name in dir(ctype):
            f = getattr(ctype, name)
            if isinstance(f, ctypes.CField) and not f.is_anonymous:
                for path, off, size in _paths(f.type, (*prefix, name)):
                    yield path, f.offset + off, size
    elif issubclass(ctype, ctypes.Array):
        elem, length = getattr(ctype, "_type_"), getattr(ctype, "_length_")
        esize = ctypes.sizeof(elem)
        for i in range(length):
            for path, off, size in _paths(elem, (*prefix, i)):
                yield path, i * esize + off, size
    else:
        yield prefix, 0, ctypes.sizeof(ctype)


def _walk(view, path):
    for step in path:
        view = view[step] if isinstance(step, int) else getattr(view, step)
    return view


@pytest.mark.parametrize("ctype", [Point, Packed, Inner, Num, Outer, Derived])
def test_offsets_match_ctypes(ctype):
    base = typed(a2 + 100, ctype)
    leaves = list(_paths(ctype))
    assert leaves
    for path, off, size in leaves:
        assert _walk(base, path) == M(a2, 100 + off, size), path


def test_layout_details():
    assert typed(a0, Packed).b == M(a0, 1, 8)
    assert typed(a0, Packed).c == M(a0, 9, 2)
    o = typed(a0, Outer)
    # Anonymous members are reachable directly, at ctypes' offsets.
    assert o.lo == M(a0, Outer.anon.offset, 2)
    assert o.hi == M(a0, Outer.anon.offset + 8, 8)
    assert isinstance(o.anon, Typed)
    # Union members share an offset.
    assert o.num.i.disp == o.num.f.disp == o.num.b[0].disp == Outer.num.offset
    assert o.num.b[7] == M(a0, Outer.num.offset + 7, 1)
    # Nested arrays and arrays of structures.
    assert isinstance(o.grid, TypedArray) and isinstance(o.grid[1], TypedArray)
    assert len(o.grid) == 2 and len(o.grid[0]) == 3
    assert o.grid[1][2] == M(a0, Outer.grid.offset + 12 + 8, 4)
    assert o.pts[2].v[3] == M(a0, Outer.pts.offset + 2 * 56 + 24 + 24, 8)
    # Pointer fields are plain 8 byte scalars, never dereferenced.
    assert o.ptr == M(a0, Outer.ptr.offset, 8)
    assert o.name == M(a0, Outer.name.offset, 8)
    # Derived structures see the base fields.
    assert typed(a0, Derived).extra == M(a0, ctypes.sizeof(Point), 2)
    assert typed(a0, Derived).v[0] == M(a0, 24, 8)


def test_bases():
    assert typed(sp, Point).y == M(sp, 4, 4)
    assert typed(sp + 16, Point).y == M(sp, 20, 4)
    assert typed(a1 - 8, Point).v[0] == M(a1, 16, 8)
    assert typed(mem[a1 + 8], Point).y == M(a1, 12, 4)
    assert typed(typed(a1, Outer).inner.addr, Inner).hi == M(a1, Outer.inner.offset + 8, 8)
    assert typed(zero + 64, ctypes.c_int64) == M(zero, 64, 8)


def test_access_width_is_checked():
    p = typed(a0, Point)
    assert enc(lw, a1, p.x) == enc(lw, a1, mem[a0])
    assert enc(lwu, a1, p.y) == enc(lwu, a1, mem[a0 + 4])
    assert enc(lb, a1, p.tag) == enc(lb, a1, mem[a0 + 8])
    assert enc(fld, fa0, p.v[1]) == enc(fld, fa0, mem[a0 + 32])
    assert enc(sd, a1, p.next) == enc(sd, a1, mem[a0 + 16])
    assert enc(flw, fa0, typed(a0, ctypes.c_float)) == enc(flw, fa0, mem[a0])
    msg = r"\(a1, 0\(a0\)\): typed\(\) operand 0\(a0\) holds 4 bytes, this access is 8 bytes$"
    with pytest.raises(EncodeError, match=msg):
        enc(ld, a1, p.x)
    with pytest.raises(EncodeError, match="holds 8 bytes, this access is 4 bytes$"):
        enc(flw, fa0, p.v[0])
    with pytest.raises(EncodeError, match="holds 1 bytes, this access is 2 bytes$"):
        enc(sh, a1, p.tag)


def _mem_mnemonics():
    """(mnemonic, template alternative, access width) of every alternative
    that takes a memory operand: loads, stores, F/D, lr/sc and the AMOs."""
    out = []
    for key, tpl in sorted(MAP_OP.items()):
        name = key.rpartition("_")[0]
        for alt in tpl.split("|"):
            if set(alt[8:]) & set("LSA"):
                op = int(alt[:8], 16)
                out.append((name, alt, None if op & 0x7F == 0x67 else 1 << ((op >> 12) & 3)))
    return out


_WIDTH = {
    "lb": 1, "lbu": 1, "sb": 1, "lh": 2, "lhu": 2, "sh": 2, "lw": 4, "lwu": 4, "sw": 4, "flw": 4, "fsw": 4,
    "ld": 8, "sd": 8, "fld": 8, "fsd": 8,
}  # fmt: skip


def test_every_memory_alternative_is_width_checked():
    alts = _mem_mnemonics()
    names = {n for n, _, _ in alts}
    # Every load, store, lr/sc and AMO, and jalr (an address, no access).
    assert {"jalr", "lr.w", "sc.d.aqrl", "amomaxu.w.rl", *_WIDTH} <= names
    assert len(names) == len(_WIDTH) + 1 + 2 * 4 * 11
    sized = {1: ctypes.c_uint8, 2: ctypes.c_uint16, 4: ctypes.c_uint32, 8: ctypes.c_uint64}
    for name, alt, width in alts:
        fn = MNEMONICS[name]
        if name in _WIDTH:
            assert width == _WIDTH[name], name
        else:
            assert width == (4 if ".w" in name else 8) or name == "jalr", name
        regs = [fa1 if c in "dm" and name.startswith("f") else a1 for c in alt[8:] if c in "DNMdm"]
        if name == "jalr":
            with pytest.raises(EncodeError, match="jalr jumps to the address of typed"):
                enc(fn, a1, typed(a0, ctypes.c_uint64))
            continue
        good = typed(a0, sized[width])
        assert enc(fn, *regs, good) == enc(fn, *regs, mem[a0]), name
        for size, ctype in sized.items():
            if size != width:
                with pytest.raises(EncodeError, match=f"holds {size} bytes, this access is {width} bytes$"):
                    enc(fn, *regs, typed(a0, ctype))


def test_jalr_rejects_a_typed_field():
    class Ops(ctypes.Structure):
        _fields_ = [("n", ctypes.c_int64), ("fn", ctypes.CFUNCTYPE(None))]

    ops = typed(a0, Ops)
    msg = (
        r"jalr: no encoding for \(ra, 8\(a0\)\): jalr jumps to the address of typed\(\) operand 8\(a0\) "
        r"and does not load it; load the field into a register and jalr that register$"
    )
    with pytest.raises(EncodeError, match=msg):
        enc(jalr, ra, ops.fn)
    # What the message says: load, then jump.
    a = Assembler(R)
    with a:
        ld(t0, ops.fn)
        jalr(t0)
    assert bytes(a.cur.buf) == enc(ld, t0, mem[a0 + 8]) + enc(jalr, t0)
    # The unsized address is still a jump target, as mem[...] is.
    assert enc(jalr, ra, ops.addr) == enc(jalr, ra, mem[a0])


def test_offsets_out_of_range():
    class Big(ctypes.Structure):
        _fields_ = [("pad", ctypes.c_uint8 * 40000), ("q", ctypes.c_int64)]

    assert enc(ld, a1, typed(a0, ctypes.c_int64 * 256)[255]) == enc(ld, a1, mem[a0 + 2040])
    assert enc(lbu, a1, typed(a0, ctypes.c_uint8 * 4096)[2047]) == enc(lbu, a1, mem[a0 + 2047])
    assert enc(lh, a1, typed(a0 - 2048, ctypes.c_int16)) == enc(lh, a1, mem[a0 - 2048])
    msg = r"ld: no encoding for \(a1, 40000\(a0\)\): offset 0x9c40 out of range \(signed 12 bit, -2048..2047\)$"
    with pytest.raises(EncodeError, match=msg):
        enc(ld, a1, typed(a0, Big).q)
    with pytest.raises(EncodeError, match=r"offset 0x800 out of range"):
        enc(lbu, a1, typed(a0, ctypes.c_uint8 * 4096)[2048])
    # Atomics take no offset; the message says how to get one.
    p = typed(a0, Point)
    msg = r"amoadd.w: no encoding for \(a1, a2, 4\(a0\)\): atomic address takes no offset, got 4; addi"
    with pytest.raises(EncodeError, match=msg):
        enc(amoadd.w, a1, a2, p.y)
    a = Assembler(R)
    with a:
        addi(t0, a0, Point.y.offset)
        amoadd.w(a1, a2, typed(t0, ctypes.c_int32))
    assert bytes(a.cur.buf) == enc(addi, t0, a0, 4) + enc(amoadd.w, a1, a2, mem[t0])
    # A field at offset 0 needs nothing.
    assert enc(amoswap.w.aq, a1, a2, p.x) == enc(amoswap.w.aq, a1, a2, mem[a0])
    assert enc(lr.w, a1, p.x) == enc(lr.w, a1, mem[a0])


def test_register_index_is_rejected():
    msg = (
        rf"^{I32}_Array_4\[a2\]: riscv64 loads and stores have no index register; put the element "
        r"address in a register first, sh2add\(t0, a2, a1\) \(Zba\) or slli\(t0, a2, 2\) and "
        rf"add\(t0, a1, t0\), and view the element with typed\(t0, ctypes.{I32}\)$"
    )
    with pytest.raises(EncodeError, match=msg):
        typed(a1, ctypes.c_int32 * 4)[a2]  # type: ignore[index]  # pyright: ignore[reportArgumentType]
    # The offset of the array goes into the view of the element.
    with pytest.raises(EncodeError, match=r"sh3add\(t0, a2, a0\) .*typed\(t0 \+ 24, ctypes.c_double\)$"):
        typed(a0, Point).v[a2]
    with pytest.raises(EncodeError, match=r"sh1add\(t0, a2, a0\) .*typed\(t0 - 8, ctypes.c_short\)$"):
        typed(a0 - 8, ctypes.c_int16 * 4)[a2]  # type: ignore[index]  # pyright: ignore[reportArgumentType]
    msg = r"^c_ubyte_Array_4\[a2\]: .* first, add\(t0, a1, a2\), and view the element with typed\(t0, ctypes.c_ubyte\)$"
    with pytest.raises(EncodeError, match=msg):
        typed(a1, ctypes.c_uint8 * 4)[a2]  # type: ignore[index]  # pyright: ignore[reportArgumentType]
    # Other element sizes multiply.
    msg = r"first, li\(t0, 56\), mul\(t0, a2, t0\) and add\(t0, a1, t0\), and view the element with typed\(t0, Point\)$"
    with pytest.raises(EncodeError, match=msg):
        typed(a1, Point * 3)[a2]  # type: ignore[index]  # pyright: ignore[reportArgumentType]
    # The temporary register is neither the base nor the index.
    with pytest.raises(EncodeError, match=rf"sh2add\(t1, t0, a1\) .*typed\(t1, ctypes.{I32}\)$"):
        typed(a1, ctypes.c_int32 * 4)[t0]  # type: ignore[index]  # pyright: ignore[reportArgumentType]
    with pytest.raises(EncodeError, match=r"li\(t2, 56\), mul\(t2, t1, t2\) and add\(t2, t0, t2\)"):
        typed(t0, Point * 3)[t1]  # type: ignore[index]  # pyright: ignore[reportArgumentType]
    # Zero size elements are all at the start of the array.
    msg = (
        r"^c_ubyte_Array_0_Array_4\[a2\]: the elements are 0 bytes, so every element is at the start of the "
        r"array and the index selects nothing; view the element with typed\(a1, ctypes.c_ubyte \* 0\)$"
    )
    with pytest.raises(EncodeError, match=msg):
        typed(a1, (ctypes.c_uint8 * 0) * 4)[a2]  # type: ignore[index]  # pyright: ignore[reportArgumentType]
    with pytest.raises(EncodeError, match=r"view the element with typed\(a1 - 8, ctypes.c_ubyte \* 0\)$"):
        typed(a1 - 8, (ctypes.c_uint8 * 0) * 4)[a2]  # type: ignore[index]  # pyright: ignore[reportArgumentType]


def build_hints(a):
    """The code the index error messages suggest, loading element a2 of
    int32 arr[4] at a1, of double v[4] in a Point at a0 and of Point pts[3]
    at a1, and element t0 of int32 arr[4] at a1."""
    with a:
        sh2add(t0, a2, a1)
        lw(a3, typed(t0, ctypes.c_int32))
        slli(t0, a2, 2)
        add(t0, a1, t0)
        lw(a3, typed(t0, ctypes.c_int32))
        sh3add(t0, a2, a0)
        fld(fa0, typed(t0 + 24, ctypes.c_double))
        add(t0, a1, a2)
        lbu(a3, typed(t0, ctypes.c_uint8))
        li(t0, 56)
        mul(t0, a2, t0)
        add(t0, a1, t0)
        lw(a3, typed(t0, Point).y)
        sh2add(t1, t0, a1)
        lw(a3, typed(t1, ctypes.c_int32))


HINTS_AS = """
sh2add t0, a2, a1
lw a3, 0(t0)
slli t0, a2, 2
add t0, a1, t0
lw a3, 0(t0)
sh3add t0, a2, a0
fld fa0, 24(t0)
add t0, a1, a2
lbu a3, 0(t0)
li t0, 56
mul t0, a2, t0
add t0, a1, t0
lw a3, 4(t0)
sh2add t1, t0, a1
lw a3, 0(t1)
"""


@requires_riscv64_oracle
def test_index_hints_match_gnu_as():
    a = Assembler(R)
    build_hints(a)
    assert bytes(a.cur.buf) == assemble(HINTS_AS, arch="riscv64")


def test_attribute_names_shadowed_by_view_properties():
    class Buf(ctypes.Structure):
        _fields_ = [("addr", ctypes.c_void_p), ("size", ctypes.c_uint32), ("ctype", ctypes.c_uint8)]

    b = typed(a0, Buf)
    assert b.size == ctypes.sizeof(Buf)
    assert b["addr"] == M(a0, 0, 8)
    assert b["size"] == M(a0, 8, 4)
    assert b["ctype"] == M(a0, 12, 1)


def test_underscore_fields_and_dir():
    class Padded(ctypes.Structure):
        _fields_ = [("_pad", ctypes.c_uint32), ("addr", ctypes.c_void_p), ("_n", ctypes.c_int16)]

    v = typed(a0, Padded)
    assert v._pad == M(a0, 0, 4)
    assert v._n == M(a0, 16, 2)
    assert v["addr"] == M(a0, 8, 8)
    names = dir(v)
    assert len(names) == len(set(names))
    assert set(names) == {"_pad", "_n", "addr", "ctype", "size"}
    with pytest.raises(AttributeError, match="fields: _pad, addr, _n"):
        v._missing
    with pytest.raises(AttributeError):
        v.__missing__


def test_field_named_like_a_view_helper():
    class Odd(ctypes.Structure):
        _fields_ = [("n", ctypes.c_int32), ("_fields", ctypes.c_int64)]

    v = typed(a0, Odd)
    assert v._fields == M(a0, 8, 8)
    assert v["_fields"] == M(a0, 8, 8)
    assert "_fields" in dir(v)


def test_mem_keeps_the_size():
    p = typed(a0, Point)
    x = p.x
    assert mem[x] is x and mem[x].size == 4
    assert enc(lw, a1, mem[p.y]) == enc(lw, a1, mem[a0 + 4])
    with pytest.raises(EncodeError, match="holds 4 bytes, this access is 8"):
        enc(ld, a1, mem[p.x])


def test_listing_shows_plain_operands():
    from jita.tools.listing import listing

    a = Assembler(R)
    p = typed(a0, Point)
    with a:
        lw(a1, p.y)
        fsd(fa0, p.v[3])
        amoadd.w(a1, a2, p.x)
    text = listing(a)
    assert "lw a1, 4(a0)" in text
    assert "fsd fa0, 48(a0)" in text
    assert "amoadd.w a1, a2, (a0)" in text


def test_typed_is_the_function():
    assert isinstance(R.typed, types.FunctionType)
    from jita.riscv64 import typed as t

    assert t is R.typed
    assert "typed" in R.__all__


def test_errors():
    with pytest.raises(EncodeError, match="bit field"):
        typed(a0, Bits).f
    assert typed(a0, Bits).n == M(a0, 0, 4)

    class Long(ctypes.Structure):
        _fields_ = [("ld", ctypes.c_longdouble)]

    if ctypes.sizeof(ctypes.c_longdouble) == 8:
        assert typed(a0, Long).ld == M(a0, 0, 8)
    else:
        with pytest.raises(EncodeError, match="long double"):
            typed(a0, Long).ld
    with pytest.raises(AttributeError, match="fields: x, y, tag, next, v"):
        typed(a0, Point).z
    with pytest.raises(AttributeError):
        typed(a0, Point)["_fields_"]
    with pytest.raises(IndexError):
        typed(a0, Point).v[4]
    with pytest.raises(IndexError):
        typed(a0, Point).v[-1]
    with pytest.raises(TypeError):
        typed(a0, Point).v["x"]
    with pytest.raises(EncodeError, match=r"c_double_Array_4\[fa0\]: fa0 cannot be an index register$"):
        typed(a0, Point).v[fa0]
    # Base and type checks.
    with pytest.raises(EncodeError, match=r"typed\(\) needs an unsized address, got a 4 byte 0\(a0\)$"):
        typed(typed(a0, Point).x, Point)
    with pytest.raises(EncodeError, match="fa0 cannot be a memory base"):
        typed(fa0, Point)  # type: ignore[call-overload]  # pyright: ignore[reportCallIssue, reportArgumentType]
    with pytest.raises(EncodeError, match="no index register"):
        typed(a0 + a1, Point)  # type: ignore[operator]  # pyright: ignore[reportOperatorIssue]
    with pytest.raises(TypeError):
        typed(0x1000, Point)  # type: ignore[call-overload]  # pyright: ignore[reportCallIssue, reportArgumentType]
    with pytest.raises(TypeError):
        typed(a0, int)
    with pytest.raises(TypeError):
        typed(a0, Point()).x  # type: ignore[call-overload]  # pyright: ignore[reportCallIssue, reportArgumentType]
    with pytest.raises(AttributeError):
        typed(a0, Point).x = 1
    with pytest.raises(EncodeError, match="size"):
        MemExpr(a0, size=3)
    with pytest.raises(EncodeError, match="size"):
        MemExpr(a0, size=True)

    class Big(ctypes.BigEndianStructure):
        _fields_ = [("n", ctypes.c_int32)]

    with pytest.raises(EncodeError, match="big endian"):
        typed(a0, Big).n


def test_flexible_array_member_is_not_bounds_checked():
    class Vec(ctypes.Structure):
        _fields_ = [("n", ctypes.c_int64), ("items", ctypes.c_int64 * 0)]

    assert typed(a0, Vec).items[5] == M(a0, 48, 8)


class Counter(ctypes.Structure):
    _fields_ = [("hits", ctypes.c_int32), ("pad", ctypes.c_int32), ("total", ctypes.c_int64)]


def build_routine(a):
    """int64 f(Point *p, int64 *vals, uint64 i, Counter *c):
    p->y = p->x + p->y; p->v[p->tag] *= 2; p->next = &p->v[3];
    vals[i] = -vals[i]; c->hits += 1 and c->total += p->y atomically;
    return p->x (sign extended).
    """
    p = typed(a0, Point)
    vals = typed(a1, ctypes.c_int64 * 0)
    c = typed(a3, Counter)
    with a:
        lw(t1, p.x)
        lw(t2, p.y)
        addw(t2, t1, t2)
        sw(t2, p.y)
        lbu(t3, p.tag)
        sh3add(t4, t3, a0)
        v = typed(t4 + Point.v.offset, ctypes.c_double)
        fld(fa0, v)
        fadd.d(fa0, fa0, fa0)
        fsd(fa0, v)
        addi(t5, a0, Point.v.offset + 3 * 8)
        sd(t5, p.next)
        sh3add(t4, a2, a1)
        ld(t5, typed(t4, ctypes.c_int64))
        neg(t5, t5)
        sd(t5, typed(t4, ctypes.c_int64))
        li(t6, 1)
        amoadd.w(zero, t6, c.hits)
        addi(t4, a3, Counter.total.offset)
        amoadd.d.aqrl(zero, t2, typed(t4, ctypes.c_int64))
        lw(a0, p.x)
        ret()
    assert isinstance(vals, TypedArray)


ROUTINE_AS = """
lw t1, 0(a0)
lw t2, 4(a0)
addw t2, t1, t2
sw t2, 4(a0)
lbu t3, 8(a0)
sh3add t4, t3, a0
fld fa0, 24(t4)
fadd.d fa0, fa0, fa0
fsd fa0, 24(t4)
addi t5, a0, 48
sd t5, 16(a0)
sh3add t4, a2, a1
ld t5, 0(t4)
neg t5, t5
sd t5, 0(t4)
li t6, 1
amoadd.w zero, t6, (a3)
addi t4, a3, 8
amoadd.d.aqrl zero, t2, (t4)
lw a0, 0(a0)
ret
"""


@requires_riscv64_oracle
def test_routine_bytes_match_gnu_as():
    a = Assembler(R)
    build_routine(a)
    assert bytes(a.cur.buf) == assemble(ROUTINE_AS, arch="riscv64")


@needs_riscv64_host
def test_execute_reads_and_writes_structure():
    a = Assembler(R)
    build_routine(a)
    pt = Point(x=-5, y=12, tag=2)
    pt.v[:] = [1.0, 2.0, 3.5, 4.0]
    vals = (ctypes.c_int64 * 4)(10, 20, 30, 40)
    cnt = Counter(hits=3, total=100)
    with a.load() as mod:
        fn = mod.function(
            ctypes.c_int64,
            ctypes.POINTER(Point),
            ctypes.POINTER(ctypes.c_int64),
            ctypes.c_uint64,
            ctypes.POINTER(Counter),
        )
        assert fn(ctypes.byref(pt), vals, 2, ctypes.byref(cnt)) == -5
        assert pt.y == 7
        assert list(pt.v) == [1.0, 2.0, 7.0, 4.0]
        assert pt.next == ctypes.addressof(pt) + Point.v.offset + 3 * 8
        assert list(vals) == [10, 20, -30, 40]
        assert (cnt.hits, cnt.total) == (4, 107)


@needs_riscv64_host
def test_execute_index_hints():
    # The code of the index error messages loads the elements they name.
    a = Assembler(R)
    arr = typed(a1, ctypes.c_int32 * 4)
    with a:
        # int32 f(int32 *arr, uint64 i): arr[i] + arr[i] (both hints)
        sh2add(t0, a1, a0)
        lw(t1, typed(t0, ctypes.c_int32))
        slli(t0, a1, 2)
        add(t0, a0, t0)
        lw(t2, typed(t0, ctypes.c_int32))
        addw(a0, t1, t2)
        ret()
    assert isinstance(arr, TypedArray)
    b = Assembler(R)
    with b:
        # double g(Point *pts, uint64 i): pts[i].v[1] through li/mul/add
        li(t0, ctypes.sizeof(Point))
        mul(t0, a1, t0)
        add(t0, a0, t0)
        fld(fa0, typed(t0, Point).v[1])
        ret()
    ints = (ctypes.c_int32 * 4)(5, 6, 7, 8)
    pts = (Point * 3)()
    pts[2].v[1] = 2.5
    with a.load() as mod, b.load() as mod2:
        f = mod.function(ctypes.c_int32, ctypes.POINTER(ctypes.c_int32), ctypes.c_uint64)
        assert [f(ints, i) for i in range(4)] == [10, 12, 14, 16]
        g = mod2.function(ctypes.c_double, ctypes.POINTER(Point), ctypes.c_uint64)
        assert g(pts, 2) == 2.5
