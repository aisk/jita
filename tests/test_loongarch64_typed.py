"""typed() on loongarch64: ctypes structures as views producing sized
memory operands, which stand for the base and offset (or base and index)
operands of the loads, stores and atomics."""

import ctypes
import platform
import types

import pytest
from oracle import assemble, requires_loongarch64_oracle

import jita.loongarch64 as L
from jita import Assembler, EncodeError
from jita.loongarch64 import *  # noqa: F403
from jita.loongarch64.insns import MNEMONICS
from jita.loongarch64.structs import Typed, TypedArray
from jita.loongarch64.table import MAP_OP

needs_loongarch64_host = pytest.mark.skipif(
    platform.machine().lower() != "loongarch64" or platform.system() == "Windows",
    reason="needs a loongarch64 POSIX host",
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


def M(base, disp=0, size=None, index=None):
    return MemExpr(base, disp, index, size)


def enc(fn, *ops):
    a = Assembler(L)
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
    assert typed(a1, ctypes.c_uint8 * 16)[a2] == M(a1, 0, 1, a2)
    assert p.size == ctypes.sizeof(Point) and p.ctype is Point
    assert p.addr == MemExpr(a0)
    assert str(p.v[1]) == "[$a0, 32]"
    assert str(p.x) == "[$a0]"
    assert str(typed(a1, ctypes.c_uint8 * 16)[a2]) == "[$a1, $a2]"
    # The operand stands for the base and offset operands.
    assert enc(ld.d, a1, p.v[1]) == enc(ld.d, a1, a0, 32)


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
    assert typed(16 + sp, Point).y == M(sp, 20, 4)
    assert typed(a1 - 8, Point).v[0] == M(a1, 16, 8)
    assert typed(a1 + 8 - 16, Point).x == M(a1, -8, 4)
    assert str(a1 + 8) == "$a1 + 8" and str(a1 - 8) == "$a1 - 8"
    assert typed(typed(a1, Outer).inner.addr, Inner).hi == M(a1, Outer.inner.offset + 8, 8)
    assert typed(zero + 64, ctypes.c_int64) == M(zero, 64, 8)
    # A register indexed element keeps its index through 1 byte elements.
    one = typed(a1, (ctypes.c_uint8 * 1) * 8)[a2]
    assert one[0] == M(a1, 0, 1, a2)
    assert typed(MemExpr(a1, index=a2), ctypes.c_uint8) == M(a1, 0, 1, a2)


def test_register_index():
    b = typed(a1, ctypes.c_uint8 * 16)
    s = typed(a1, ctypes.c_int8 * 16)
    assert b[a2] == M(a1, 0, 1, a2)
    assert b[zero] == M(a1, 0, 1, zero)
    assert enc(ldx.bu, a0, b[a2]) == enc(ldx.bu, a0, a1, a2)
    assert enc(ldx.b, a0, s[a2]) == enc(ldx.b, a0, a1, a2)
    assert enc(stx.b, a0, b[a2]) == enc(stx.b, a0, a1, a2)
    assert enc(preldx, 0, b[a2]) == enc(preldx, 0, a1, a2)
    # 1 byte structures and unions: a field at offset 0.
    class Flags(ctypes.Union):
        _fields_ = [("all", ctypes.c_uint8), ("sign", ctypes.c_int8)]

    f = typed(a1, Flags * 8)[a2]
    assert f.all == f.sign == M(a1, 0, 1, a2)


def test_access_width_is_checked():
    p = typed(a0, Point)
    assert enc(ld.w, a1, p.x) == enc(ld.w, a1, a0, 0)
    assert enc(ld.wu, a1, p.y) == enc(ld.wu, a1, a0, 4)
    assert enc(ld.b, a1, p.tag) == enc(ld.b, a1, a0, 8)
    assert enc(fld.d, fa0, p.v[1]) == enc(fld.d, fa0, a0, 32)
    assert enc(st.d, a1, p.next) == enc(st.d, a1, a0, 16)
    assert enc(fld.s, fa0, typed(a0, ctypes.c_float)) == enc(fld.s, fa0, a0, 0)
    msg = r"^ld.d: no encoding for \(a1, \[\$a0\]\): typed\(\) operand \[\$a0\] holds 4 bytes, this access is 8 bytes$"
    with pytest.raises(EncodeError, match=msg):
        enc(ld.d, a1, p.x)
    with pytest.raises(EncodeError, match="holds 8 bytes, this access is 4 bytes$"):
        enc(fld.s, fa0, p.v[0])
    with pytest.raises(EncodeError, match="holds 1 bytes, this access is 2 bytes$"):
        enc(st.h, a1, p.tag)
    # The unsized address of a view has no width to check.
    assert enc(ld.d, a1, p.addr) == enc(ld.d, a1, a0, 0)
    assert enc(ld.b, a1, p.v.addr) == enc(ld.b, a1, a0, 24)


_WIDTH = {"b": 1, "bu": 1, "h": 2, "hu": 2, "w": 4, "wu": 4, "s": 4, "d": 8, "du": 8}


def _mem_alternatives():
    """(mnemonic, template alternative) of every alternative that takes a
    memory operand."""
    return [
        (key.rpartition("_")[0], alt)
        for key, tpl in sorted(MAP_OP.items())
        for alt in tpl.split("|")
        if set(alt[8:]) & set("mnrb")
    ]


def test_every_memory_alternative_is_width_checked():
    alts = _mem_alternatives()
    names = {n for n, _ in alts}
    heads = {n.split(".")[0] for n in names}
    assert heads == {
        "ld", "st", "fld", "fst", "preld", "ldx", "stx", "fldx", "fstx", "preldx", "ldptr", "stptr", "ll", "sc",
        "ldgt", "ldle", "stgt", "stle", "fldgt", "fldle", "fstgt", "fstle",
        "amswap", "amadd", "amand", "amor", "amxor", "ammax", "ammin",
        "amswap_db", "amadd_db", "amand_db", "amor_db", "amxor_db", "ammax_db", "ammin_db",
    }  # fmt: skip
    sized = {1: ctypes.c_uint8, 2: ctypes.c_uint16, 4: ctypes.c_uint32, 8: ctypes.c_uint64}
    for name, alt in alts:
        fn = MNEMONICS[name]
        width = _WIDTH[name.rpartition(".")[2]] if "." in name else None

        def ops(m, alt=alt):
            out = []
            for c in alt[8:]:
                if c in "mnrb":
                    out.append(m)
                elif c == "D":
                    out.append(a4)
                elif c == "K":
                    out.append(a5)
                elif c == "d":
                    out.append(fa1)
                elif c == "P":
                    out.append(0)
            return out

        def view(ctype, alt=alt):
            if "r" in alt[8:]:
                return typed(a1, ctypes.c_uint8 * 4)[a2] if ctype is ctypes.c_uint8 else None
            return typed(a1, ctype)

        plain = MemExpr(a1, index=a2) if "r" in alt[8:] else MemExpr(a1)
        if width is None:  # preld, preldx: any size
            for ctype in sized.values():
                if (m := view(ctype)) is not None:
                    assert enc(fn, *ops(m)) == enc(fn, *ops(plain)), name
            continue
        good = view(sized[width])
        if good is not None:
            assert enc(fn, *ops(good)) == enc(fn, *ops(plain)), name
        for size, ctype in sized.items():
            bad = view(ctype)
            if size != width and bad is not None:
                with pytest.raises(EncodeError, match=f"holds {size} bytes, this access is {width} bytes$"):
                    enc(fn, *ops(bad))


def test_forms_must_match():
    p = typed(a0, Point)
    b = typed(a1, ctypes.c_uint8 * 16)
    # An indexed element needs the indexed instruction, and back.
    msg = (
        r"^ld.bu: no encoding for \(a3, \[\$a1, \$a2\]\): "
        r"ld.bu cannot take the register indexed operand \[\$a1, \$a2\]; use ldx.bu$"
    )
    with pytest.raises(EncodeError, match=msg):
        enc(ld.bu, a3, b[a2])
    msg = r"stptr.d cannot take the register indexed operand \[\$a1, \$a2\]; use stx.d$"
    with pytest.raises(EncodeError, match=msg):
        enc(stptr.d, a3, MemExpr(a1, index=a2))
    msg = (
        r"^ldx.w: no encoding for \(a3, \[\$a0, 4\]\): "
        r"ldx.w takes a register indexed operand, \[\$a0, 4\] has no index; use ld.w$"
    )
    with pytest.raises(EncodeError, match=msg):
        enc(ldx.w, a3, p.y)
    msg = r"preldx takes a register indexed operand, \[\$a0, 4\] has no index; use preld$"
    with pytest.raises(EncodeError, match=msg):
        enc(preldx, 0, p.y)
    msg = r"fstx.d takes a register indexed operand, \[\$a0, 24\] has no index; use fst.d$"
    with pytest.raises(EncodeError, match=msg):
        enc(fstx.d, fa0, p.v[0])
    assert enc(ld.w, a3, p.y) == enc(ld.w, a3, a0, 4)
    assert enc(fst.d, fa0, p.v[0]) == enc(fst.d, fa0, a0, 24)
    # ll, sc, am* and ldgt have no indexed form.
    msg = (
        r"ll.w cannot take the register indexed operand \[\$a1, \$a2\]; add.d the index to the base into a "
        r"register first and view the element there$"
    )
    with pytest.raises(EncodeError, match=msg):
        enc(ll.w, a3, MemExpr(a1, index=a2, size=4))
    a = Assembler(L)
    with a:
        add.d(t0, a1, a2)
        ll.w(a3, typed(t0, ctypes.c_int32))
    assert bytes(a.cur.buf) == enc(add.d, t0, a1, a2) + enc(ll.w, a3, t0, 0)
    # am* and the bounds checked loads take a bare base.
    msg = (
        r"^amadd_db.w: no encoding for \(a3, a4, \[\$a0, 4\]\): amadd_db.w takes a bare base register, "
        r"\[\$a0, 4\] is 4 bytes past \$a0; addi.d the offset into a register first and view the field there$"
    )
    with pytest.raises(EncodeError, match=msg):
        enc(amadd_db.w, a3, a4, p.y)
    with pytest.raises(EncodeError, match=r"ldgt.d takes a bare base register, \[\$a0, 16\] is 16 bytes past"):
        enc(ldgt.d, a3, p.next, a4)
    a = Assembler(L)
    with a:
        addi.d(t0, a0, Point.y.offset)
        amadd_db.w(a3, a4, typed(t0, ctypes.c_int32))
        addi.d(t0, a0, Point.next.offset)
        ldgt.d(a3, typed(t0, ctypes.c_void_p), a4)
    want = enc(addi.d, t0, a0, 4) + enc(amadd_db.w, a3, a4, t0) + enc(addi.d, t0, a0, 16) + enc(ldgt.d, a3, t0, a4)
    assert bytes(a.cur.buf) == want
    assert enc(amswap.w, a3, a4, p.x) == enc(amswap.w, a3, a4, a0)
    assert enc(fldle.d, fa0, typed(a0, ctypes.c_double), a4) == enc(fldle.d, fa0, a0, a4)
    # am* rd must still differ from the base.
    with pytest.raises(EncodeError, match="rd must differ from rk and rj"):
        enc(amadd.w, a0, a4, p.x)
    # An address is not an operand; write its base and offset.
    msg = (
        r"^ld.w: no encoding for \(a3, \$a0 \+ 8\): "
        r"\$a0 \+ 8 is an address for typed\(\), not an operand; write a0, 8$"
    )
    with pytest.raises(EncodeError, match=msg):
        enc(ld.w, a3, a0 + 8)
    assert enc(ld.w, a3, a0, 8) == enc(ld.w, a3, typed(a0 + 8, ctypes.c_int32))
    with pytest.raises(EncodeError, match=r"ldx.w adds an index register, write a1, index$"):
        enc(ldx.w, a0, a1 + 8)
    with pytest.raises(EncodeError, match=r"amadd.w takes a bare base, so addi.d the offset into a register first$"):
        enc(amadd.w, a0, a2, a1 + 8)
    with pytest.raises(EncodeError, match=r"not an operand; write a1$"):
        enc(amadd.w, a0, a2, a1 + 0)
    msg = r"expected a memory operand .*; for the address in a4 write st.w\(a3, a4, 0\)$"
    with pytest.raises(EncodeError, match=msg):
        enc(st.w, a3, a4)


def test_offsets_out_of_range():
    class Big(ctypes.Structure):
        _fields_ = [("pad", ctypes.c_uint8 * 40000), ("q", ctypes.c_int64)]

    class Mid(ctypes.Structure):
        _fields_ = [("pad", ctypes.c_uint8 * 3000), ("q", ctypes.c_int64), ("w", ctypes.c_int32 * 2)]

    assert enc(ld.d, a1, typed(a0, ctypes.c_int64 * 256)[255]) == enc(ld.d, a1, a0, 2040)
    assert enc(ld.bu, a1, typed(a0 - 2048, ctypes.c_uint8)) == enc(ld.bu, a1, a0, -2048)
    msg = (
        r"^ld.d: no encoding for \(a1, \[\$a0, 3000\]\): offset 0xbb8 out of range "
        r"\(signed 12 bit, -0x800..0x7ff\); use ldptr.d$"
    )
    with pytest.raises(EncodeError, match=msg):
        enc(ld.d, a1, typed(a0, Mid).q)
    with pytest.raises(EncodeError, match=r"offset 0xbc4 out of range \(signed 12 bit, -0x800..0x7ff\); use stptr.w$"):
        enc(st.w, a1, typed(a0, Mid).w[1])
    # Only when ldptr or stptr encodes the offset: a multiple of 4 in range.
    with pytest.raises(EncodeError, match=r"offset 0xbba out of range \(signed 12 bit, -0x800..0x7ff\)$"):
        enc(ld.d, a1, typed(a0 + 3002, ctypes.c_int64))
    with pytest.raises(EncodeError, match=r"offset 0x9c40 out of range \(signed 12 bit, -0x800..0x7ff\)$"):
        enc(st.d, a1, typed(a0, Big).q)
    # ld.wu, ld.h and fld have no ldptr form.
    with pytest.raises(EncodeError, match=r"offset 0xbc4 out of range \(signed 12 bit, -0x800..0x7ff\)$"):
        enc(ld.wu, a1, typed(a0 + 3012, ctypes.c_uint32))
    with pytest.raises(EncodeError, match=r"offset 0xbb8 out of range \(signed 12 bit, -0x800..0x7ff\)$"):
        enc(fld.d, fa0, typed(a0 + 3000, ctypes.c_double))
    # ldptr, stptr, ll and sc reach further, in multiples of 4.
    assert enc(ldptr.d, a1, typed(a0, Mid).q) == enc(ldptr.d, a1, a0, 3000)
    assert enc(stptr.w, a1, typed(a0, Mid).w[1]) == enc(stptr.w, a1, a0, 3012)
    assert enc(ll.d, a1, typed(a0 + 32764, ctypes.c_int64)) == enc(ll.d, a1, a0, 32764)
    assert enc(sc.w, a1, typed(a0 - 32768, ctypes.c_int32)) == enc(sc.w, a1, a0, -32768)
    msg = r"^ldptr.d: no encoding for \(a1, \[\$a0, 40000\]\): offset 0x9c40 out of range \(-0x8000..0x7ffc\)$"
    with pytest.raises(EncodeError, match=msg):
        enc(ldptr.d, a1, typed(a0, Big).q)
    with pytest.raises(EncodeError, match=r"offset 0x8000 out of range \(-0x8000..0x7ffc\)$"):
        enc(sc.d, a1, typed(a0 + 32768, ctypes.c_int64))
    # A misaligned offset in reach of ld and st, but ll and sc have no
    # such form.
    with pytest.raises(EncodeError, match=r"offset 2 is not a multiple of 4; use ld.w$"):
        enc(ldptr.w, a1, typed(a0 + 2, ctypes.c_int32))
    with pytest.raises(EncodeError, match=r"offset -0x7fe is not a multiple of 4; use st.d$"):
        enc(stptr.d, a1, typed(a0 - 2046, ctypes.c_int64))
    with pytest.raises(EncodeError, match=r"offset 0xbba is not a multiple of 4$"):
        enc(ldptr.d, a1, typed(a0 + 3002, ctypes.c_int64))
    with pytest.raises(EncodeError, match=r"offset 2 is not a multiple of 4$"):
        enc(ll.w, a1, typed(a0 + 2, ctypes.c_int32))
    with pytest.raises(EncodeError, match=r"offset 6 is not a multiple of 4$"):
        enc(sc.d, a1, typed(a0 + 6, ctypes.c_int64))
    # The suggested instruction encodes the operand.
    assert enc(ld.w, a1, typed(a0 + 2, ctypes.c_int32)) == enc(ld.w, a1, a0, 2)
    assert enc(st.d, a1, typed(a0 - 2046, ctypes.c_int64)) == enc(st.d, a1, a0, -2046)


def test_register_index_hints():
    msg = (
        rf"^{I32}_Array_4\[a2\]: ldx adds the index unscaled, and the elements are 4 bytes; put the element "
        rf"address in a register first, alsl.d\(t0, a2, a1, 2\), and view the element with typed\(t0, ctypes.{I32}\)$"
    )
    with pytest.raises(EncodeError, match=msg):
        typed(a1, ctypes.c_int32 * 4)[a2]
    msg = r"alsl.d\(t0, a2, a0, 3\), and view the element with typed\(t0 \+ 24, ctypes.c_double\)$"
    with pytest.raises(EncodeError, match=msg):
        typed(a0, Point).v[a2]
    with pytest.raises(EncodeError, match=r"alsl.d\(t0, a2, a1, 4\), and view the element with typed\(t0, Inner\)$"):
        typed(a1, Inner * 4)[a2]
    msg = (
        r"^Point_Array_3\[a2\]: ldx adds the index unscaled, and the elements are 56 bytes; put the element "
        r"address in a register first, li.d\(t0, 56\), mul.d\(t0, a2, t0\) and add.d\(t0, a1, t0\), and view the "
        r"element with typed\(t0 - 8, Point\)$"
    )
    with pytest.raises(EncodeError, match=msg):
        typed(a1 - 8, Point * 3)[a2]
    # 1 byte elements, but the array is not at the base.
    msg = (
        r"^c_ubyte_Array_8\[a2\]: the array is at \[\$a0, 8\], and ldx takes no offset besides the index "
        r"register; put the element address in a register first, add.d\(t0, a0, a2\), and view the element "
        r"with typed\(t0 \+ 8, ctypes.c_ubyte\)$"
    )
    with pytest.raises(EncodeError, match=msg):
        typed(a0, Num * 2)[1].b[a2]
    # The temporary register is neither the base nor the index.
    with pytest.raises(EncodeError, match=r"alsl.d\(t1, t0, a1, 2\)"):
        typed(a1, ctypes.c_int32 * 4)[t0]
    with pytest.raises(EncodeError, match=r"li.d\(t2, 56\), mul.d\(t2, t1, t2\) and add.d\(t2, t0, t2\)"):
        typed(t0, Point * 3)[t1]
    # Zero size elements are all at the start of the array.
    msg = (
        r"^c_ubyte_Array_0_Array_4\[a2\]: the elements are 0 bytes, so every element is at the start of the "
        r"array and the index selects nothing; view the element with typed\(a1, ctypes.c_ubyte \* 0\)$"
    )
    with pytest.raises(EncodeError, match=msg):
        typed(a1, (ctypes.c_uint8 * 0) * 4)[a2]
    with pytest.raises(EncodeError, match=r"view the element with typed\(a1 \+ 8, ctypes.c_ubyte \* 0\)$"):
        typed(a1 + 8, (ctypes.c_uint8 * 0) * 4)[a2]


def build_hints(a):
    """The code the index error messages suggest: element a2 of int32
    arr[4] at a1, of double v[4] in a Point at a0, of Point pts[3] at
    a1 - 8, of uint8 b[8] at a0 + 8, and element t0 of int32 arr[4] at a1."""
    with a:
        alsl.d(t0, a2, a1, 2)
        ld.w(a3, typed(t0, ctypes.c_int32))
        alsl.d(t0, a2, a0, 3)
        fld.d(fa0, typed(t0 + 24, ctypes.c_double))
        li.d(t0, 56)
        mul.d(t0, a2, t0)
        add.d(t0, a1, t0)
        ld.w(a3, typed(t0 - 8, Point).y)
        add.d(t0, a0, a2)
        ld.bu(a3, typed(t0 + 8, ctypes.c_uint8))
        alsl.d(t1, t0, a1, 2)
        ld.w(a3, typed(t1, ctypes.c_int32))


HINTS_AS = """
alsl.d $t0, $a2, $a1, 2
ld.w $a3, $t0, 0
alsl.d $t0, $a2, $a0, 3
fld.d $fa0, $t0, 24
li.d $t0, 56
mul.d $t0, $a2, $t0
add.d $t0, $a1, $t0
ld.w $a3, $t0, -4
add.d $t0, $a0, $a2
ld.bu $a3, $t0, 8
alsl.d $t1, $t0, $a1, 2
ld.w $a3, $t1, 0
"""


@requires_loongarch64_oracle
def test_index_hints_match_gnu_as():
    a = Assembler(L)
    build_hints(a)
    assert bytes(a.cur.buf) == assemble(HINTS_AS, arch="loongarch64")


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


def test_listing_shows_the_operands():
    from jita.tools.listing import listing

    a = Assembler(L)
    p = typed(a0, Point)
    with a:
        ld.w(a1, p.y)
        fst.d(fa0, p.v[3])
        ldx.bu(a1, typed(a2, ctypes.c_uint8 * 4)[a3])
        amadd_db.w(a1, a2, p.x)
        ldptr.d(a1, p.next)
    text = listing(a)
    assert "ld.w $a1, $a0, 4" in text
    assert "fst.d $fa0, $a0, 48" in text
    assert "ldx.bu $a1, $a2, $a3" in text
    assert "amadd_db.w $a1, $a2, $a0" in text
    assert "ldptr.d $a1, $a0, 16" in text


def test_typed_is_the_function():
    assert isinstance(L.typed, types.FunctionType)
    from jita.loongarch64 import typed as t

    assert t is L.typed
    assert {"typed", "MemExpr"} <= set(L.__all__)
    assert "Addr" not in L.__all__


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
    with pytest.raises(EncodeError, match=r"c_ubyte_Array_4\[\$fa0\]: \$fa0 cannot be an index register$"):
        typed(a0, ctypes.c_uint8 * 4)[fa0]  # type: ignore[index]  # pyright: ignore[reportArgumentType]
    # An index, then an offset or another index.
    msg = (
        r"^Inner.hi is 8 bytes past \[\$a1, \$a2\], and a loongarch64 address cannot have both an index "
        r"register and an offset; add.d the index to the base into a register first and view the element there$"
    )
    with pytest.raises(EncodeError, match=msg):
        typed(MemExpr(a1, index=a2), Inner).hi
    assert typed(MemExpr(a1, index=a2), Inner).lo == M(a1, 0, 2, a2)
    with pytest.raises(EncodeError, match=r"already has an index register"):
        typed(a1, (ctypes.c_uint8 * 1) * 4)[a2][a3]
    # Base and type checks.
    with pytest.raises(EncodeError, match=r"typed\(\) needs an unsized address, got a 4 byte \[\$a0\]$"):
        typed(typed(a0, Point).x, Point)
    with pytest.raises(EncodeError, match=r"\$fa0 cannot be a memory base"):
        typed(fa0, Point)  # type: ignore[call-overload]  # pyright: ignore[reportCallIssue, reportArgumentType]
    with pytest.raises(EncodeError, match=r"a loongarch64 address is a base register plus an offset, not \$a0 \+ \$a1"):
        typed(a0 + a1, Point)  # type: ignore[operator]  # pyright: ignore[reportOperatorIssue]
    with pytest.raises(TypeError):
        _ = fa0 + 8  # type: ignore[operator]  # pyright: ignore[reportOperatorIssue]
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
    with pytest.raises(EncodeError, match="both an index register and an offset"):
        MemExpr(a0, 8, a1)
    with pytest.raises(EncodeError, match="cannot be a memory index"):
        MemExpr(a0, index=fa1)  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]

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
    """int64 f(Point *p, int64 *vals, uint64 i, Counter *c, uint8 *bytes):
    p->y = p->x + p->y; p->v[p->tag] *= 2; p->next = &p->v[3];
    vals[i] = -vals[i]; bytes[i] += 1; c->hits += 1 and c->total += p->y
    atomically; return p->x (sign extended).
    """
    p = typed(a0, Point)
    c = typed(a3, Counter)
    b = typed(a4, ctypes.c_uint8 * 0)
    with a:
        ld.w(t1, p.x)
        ld.w(t2, p.y)
        add.w(t2, t1, t2)
        st.w(t2, p.y)
        ld.bu(t3, p.tag)
        alsl.d(t4, t3, a0, 3)
        v = typed(t4 + Point.v.offset, ctypes.c_double)
        fld.d(fa0, v)
        fadd.d(fa0, fa0, fa0)
        fst.d(fa0, v)
        addi.d(t5, a0, Point.v.offset + 3 * 8)
        stptr.d(t5, p.next)
        alsl.d(t4, a2, a1, 3)
        ld.d(t5, typed(t4, ctypes.c_int64))
        sub.d(t5, zero, t5)
        st.d(t5, typed(t4, ctypes.c_int64))
        ldx.bu(t5, b[a2])
        addi.w(t5, t5, 1)
        stx.b(t5, b[a2])
        addi.w(t6, zero, 1)
        amadd_db.w(zero, t6, c.hits)
        addi.d(t4, a3, Counter.total.offset)
        amadd_db.d(zero, t2, typed(t4, ctypes.c_int64))
        ld.w(a0, p.x)
        ret()


ROUTINE_AS = """
ld.w $t1, $a0, 0
ld.w $t2, $a0, 4
add.w $t2, $t1, $t2
st.w $t2, $a0, 4
ld.bu $t3, $a0, 8
alsl.d $t4, $t3, $a0, 3
fld.d $fa0, $t4, 24
fadd.d $fa0, $fa0, $fa0
fst.d $fa0, $t4, 24
addi.d $t5, $a0, 48
stptr.d $t5, $a0, 16
alsl.d $t4, $a2, $a1, 3
ld.d $t5, $t4, 0
sub.d $t5, $zero, $t5
st.d $t5, $t4, 0
ldx.bu $t5, $a4, $a2
addi.w $t5, $t5, 1
stx.b $t5, $a4, $a2
addi.w $t6, $zero, 1
amadd_db.w $zero, $t6, $a3
addi.d $t4, $a3, 8
amadd_db.d $zero, $t2, $t4
ld.w $a0, $a0, 0
ret
"""


@requires_loongarch64_oracle
def test_routine_bytes_match_gnu_as():
    a = Assembler(L)
    build_routine(a)
    assert bytes(a.cur.buf) == assemble(ROUTINE_AS, arch="loongarch64")


@needs_loongarch64_host
def test_execute_reads_and_writes_structure():
    a = Assembler(L)
    build_routine(a)
    pt = Point(x=-5, y=12, tag=2)
    pt.v[:] = [1.0, 2.0, 3.5, 4.0]
    vals = (ctypes.c_int64 * 4)(10, 20, 30, 40)
    cnt = Counter(hits=3, total=100)
    raw = (ctypes.c_uint8 * 4)(1, 2, 255, 4)
    with a.load() as mod:
        fn = mod.function(
            ctypes.c_int64,
            ctypes.POINTER(Point),
            ctypes.POINTER(ctypes.c_int64),
            ctypes.c_uint64,
            ctypes.POINTER(Counter),
            ctypes.POINTER(ctypes.c_uint8),
        )
        assert fn(ctypes.byref(pt), vals, 2, ctypes.byref(cnt), raw) == -5
        assert pt.y == 7
        assert list(pt.v) == [1.0, 2.0, 7.0, 4.0]
        assert pt.next == ctypes.addressof(pt) + Point.v.offset + 3 * 8
        assert list(vals) == [10, 20, -30, 40]
        assert list(raw) == [1, 2, 0, 4]
        assert (cnt.hits, cnt.total) == (4, 107)


@needs_loongarch64_host
def test_execute_index_hints():
    # The code of the index error messages loads the elements they name.
    a = Assembler(L)
    with a:
        # int32 f(int32 *arr, uint64 i): arr[i]
        alsl.d(t0, a1, a0, 2)
        ld.w(a0, typed(t0, ctypes.c_int32))
        ret()
    b = Assembler(L)
    with b:
        # double g(Point *pts, uint64 i): pts[i].v[1] through li.d/mul.d/add.d
        li.d(t0, ctypes.sizeof(Point))
        mul.d(t0, a1, t0)
        add.d(t0, a0, t0)
        fld.d(fa0, typed(t0, Point).v[1])
        ret()
    ints = (ctypes.c_int32 * 4)(5, 6, 7, 8)
    pts = (Point * 3)()
    pts[2].v[1] = 2.5
    with a.load() as mod, b.load() as mod2:
        f = mod.function(ctypes.c_int32, ctypes.POINTER(ctypes.c_int32), ctypes.c_uint64)
        assert [f(ints, i) for i in range(4)] == [5, 6, 7, 8]
        g = mod2.function(ctypes.c_double, ctypes.POINTER(Point), ctypes.c_uint64)
        assert g(pts, 2) == 2.5
