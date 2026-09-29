"""typed() on aarch64: ctypes structures as views producing sized memory
operands."""

import ctypes
import platform
import types

import pytest
from oracle import assemble, requires_aarch64_oracle

import jita.aarch64 as A
from jita import Assembler, EncodeError
from jita.aarch64 import *  # noqa: F403
from jita.aarch64.structs import Typed, TypedArray

needs_aarch64_host = pytest.mark.skipif(
    platform.machine().lower() not in ("aarch64", "arm64") or platform.system() == "Windows",
    reason="needs an aarch64 POSIX host",
)


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


def M(base, disp=0, size=None, index=None, mod=None):
    return MemExpr(base, index, mod, disp, size=size)


def enc(fn, *ops):
    a = Assembler(A)
    fn(*ops, asm=a)
    return bytes(a.cur.buf)


def test_spec_example():
    p = typed(x0, Point)
    assert p.x == M(x0, 0, 4)
    assert p.y == M(x0, 4, 4)
    assert p.tag == M(x0, 8, 1)
    assert p.next == M(x0, 16, 8)
    assert p.v[1] == M(x0, 32, 8)
    assert p.v.addr == M(x0, 24)
    assert p.v.addr.size is None
    assert typed(x2, ctypes.c_double * 4)[x3] == M(x2, 0, 8, x3, Mod("lsl", 3))
    assert p.size == ctypes.sizeof(Point) and p.ctype is Point
    assert p.addr == mem[x0]
    assert str(p.v[1]) == "[x0, #32]"


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

    assert typed(x19, S).f == M(x19, 8, size)
    assert typed(x19, ctype) == M(x19, 0, size)


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
    base = typed(x12 + 100, ctype)
    leaves = list(_paths(ctype))
    assert leaves
    for path, off, size in leaves:
        assert _walk(base, path) == M(x12, 100 + off, size), path


def test_layout_details():
    assert typed(x0, Packed).b == M(x0, 1, 8)
    assert typed(x0, Packed).c == M(x0, 9, 2)
    o = typed(x0, Outer)
    # Anonymous members are reachable directly, at ctypes' offsets.
    assert o.lo == M(x0, Outer.anon.offset, 2)
    assert o.hi == M(x0, Outer.anon.offset + 8, 8)
    assert isinstance(o.anon, Typed)
    # Union members share an offset.
    assert o.num.i.disp == o.num.f.disp == o.num.b[0].disp == Outer.num.offset
    assert o.num.b[7] == M(x0, Outer.num.offset + 7, 1)
    # Nested arrays and arrays of structures.
    assert isinstance(o.grid, TypedArray) and isinstance(o.grid[1], TypedArray)
    assert len(o.grid) == 2 and len(o.grid[0]) == 3
    assert o.grid[1][2] == M(x0, Outer.grid.offset + 12 + 8, 4)
    assert o.pts[2].v[3] == M(x0, Outer.pts.offset + 2 * 56 + 24 + 24, 8)
    with pytest.raises(EncodeError, match="scale"):
        typed(x0, (ctypes.c_int32 * 3) * 2)[x1]  # 12 byte rows
    # Pointer fields are plain 8 byte scalars, never dereferenced.
    assert o.ptr == M(x0, Outer.ptr.offset, 8)
    assert o.name == M(x0, Outer.name.offset, 8)
    # Derived structures see the base fields.
    assert typed(x0, Derived).extra == M(x0, ctypes.sizeof(Point), 2)
    assert typed(x0, Derived).v[0] == M(x0, 24, 8)


def test_bases():
    assert typed(sp, Point).y == M(sp, 4, 4)
    assert typed(sp + 16, Point).y == M(sp, 20, 4)
    assert typed(x1 - 8, Point).v[0] == M(x1, 16, 8)
    assert typed(mem[x1 + 8], Point).y == M(x1, 12, 4)
    # An address with an index, when nothing adds an offset to it.
    assert typed(x1 + (x2 << 2), Point).x == M(x1, 0, 4, x2, Mod("lsl", 2))
    assert typed(x1 + w2.uxtw(), Inner).lo == M(x1, 0, 2, w2, Mod("uxtw", None))


def test_register_index():
    arr = typed(x1, ctypes.c_int16 * 10)
    assert arr[x2] == M(x1, 0, 2, x2, Mod("lsl", 1))
    assert typed(x1, ctypes.c_uint8 * 4)[x2] == M(x1, 0, 1, x2)
    assert typed(x1, ctypes.c_int64 * 4)[x2] == M(x1, 0, 8, x2, Mod("lsl", 3))
    # A w index needs an extend; the element size sets the amount.
    assert typed(x1, ctypes.c_int32 * 4)[w2.uxtw()] == M(x1, 0, 4, w2, Mod("uxtw", 2))
    assert typed(x1, ctypes.c_int32 * 4)[w2.sxtw()] == M(x1, 0, 4, w2, Mod("sxtw", 2))
    assert typed(x1, ctypes.c_uint8 * 4)[w2.uxtw()] == M(x1, 0, 1, w2, Mod("uxtw", None))
    assert typed(x1, ctypes.c_int64 * 4)[x2.sxtx()] == M(x1, 0, 8, x2, Mod("sxtx", 3))
    # A matching explicit amount is accepted.
    assert typed(x1, ctypes.c_int32 * 4)[x2 << 2] == M(x1, 0, 4, x2, Mod("lsl", 2))
    assert typed(x1, ctypes.c_int32 * 4)[w2.sxtw(2)] == M(x1, 0, 4, w2, Mod("sxtw", 2))
    # Elements that are structures or arrays work when the field accessed
    # is at offset 0 and fills the element.
    one = typed(x1, Num * 4)[x2]
    assert one.i == one.f == M(x1, 0, 8, x2, Mod("lsl", 3))
    assert typed(x1, (ctypes.c_uint16 * 1) * 4)[x2][0] == M(x1, 0, 2, x2, Mod("lsl", 1))
    assert str(arr[x2]) == "[x1, x2, lsl #1]"


def test_access_width_is_checked():
    p = typed(x0, Point)
    assert enc(ldr, w1, p.x) == enc(ldr, w1, mem[x0])
    assert enc(ldrsw, x1, p.y) == enc(ldrsw, x1, mem[x0 + 4])
    assert enc(ldrsb, x1, p.tag) == enc(ldrsb, x1, mem[x0 + 8])
    assert enc(ldr, d0, p.v[1]) == enc(ldr, d0, mem[x0 + 32])
    assert enc(str_, x1, p.next) == enc(str_, x1, mem[x0 + 16])
    assert enc(ldr, s0, typed(x0, ctypes.c_float)) == enc(ldr, s0, mem[x0])
    for fn, ops in [
        (ldr, (x1, p.x)),
        (ldr, (d0, p.x)),
        (ldrb, (w1, p.x)),
        (ldrh, (w1, p.tag)),
        (ldrsw, (x1, p.next)),
        (str_, (w1, p.next)),
        (strb, (w1, p.y)),
        (ldr, (s0, p.v[0])),
    ]:
        with pytest.raises(EncodeError, match=r"typed\(\) operand \[x0.*\] holds \d bytes, this access is"):
            enc(fn, *ops)
    # The message names the operand as a typed one, str() stays plain.
    msg = r"\(x1, \[x0\]\): typed\(\) operand \[x0\] holds 4 bytes, this access is 8 bytes$"
    with pytest.raises(EncodeError, match=msg):
        enc(ldr, x1, p.x)
    assert str(p.x) == "[x0]"
    # ldp and stp compare the size of one register.
    ints = typed(x0, ctypes.c_int32 * 4)
    assert enc(ldp, w1, w2, ints[0]) == enc(ldp, w1, w2, mem[x0])
    assert enc(stp, d0, d1, p.v[2]) == enc(stp, d0, d1, mem[x0 + 40])
    assert enc(ldpsw, x1, x2, ints[2]) == enc(ldpsw, x1, x2, mem[x0 + 8])
    with pytest.raises(EncodeError, match="holds 4 bytes, this access is 8"):
        enc(ldp, x1, x2, ints[0])
    with pytest.raises(EncodeError, match="holds 8 bytes, this access is 16"):
        enc(ldp, q0, q1, p.v[0])
    # An index scaled by the element size is the scale the access needs.
    arr = typed(x1, ctypes.c_int16 * 10)
    assert enc(ldrh, w0, arr[x2]) == enc(ldrh, w0, mem[x1 + (x2 << 1)])
    assert enc(ldrsh, x0, arr[w2.sxtw()]) == enc(ldrsh, x0, mem[x1 + w2.sxtw(1)])
    with pytest.raises(EncodeError, match="holds 2 bytes, this access is 4"):
        enc(ldr, w0, arr[x2])


def test_offsets_out_of_range():
    class Big(ctypes.Structure):
        _fields_ = [
            ("pad", ctypes.c_uint8 * 40000),
            ("q", ctypes.c_int64),
            ("pad2", ctypes.c_uint8 * 3),
            ("odd", ctypes.c_int16),
        ]

    class Tight(ctypes.Structure):
        _pack_ = 1
        _layout_ = "ms"
        _fields_ = [("pad", ctypes.c_uint8 * 301), ("u", ctypes.c_int32)]

    b = typed(x0, Big)
    # Scaled offsets up to 4095 * size, unscaled ones in -256..255.
    assert enc(ldr, x1, typed(x0, ctypes.c_int64 * 512)[511]) == enc(ldr, x1, mem[x0 + 4088])
    with pytest.raises(EncodeError, match="out of range"):
        enc(ldr, x1, b.q)  # 40000 > 4095 * 8
    with pytest.raises(EncodeError, match="out of range"):
        enc(ldr, w1, typed(x0, Tight).u)  # 301 is neither a multiple of 4 nor below 256
    assert enc(ldr, x1, typed(x0, Packed).b) == enc(ldr, x1, mem[x0 + 1])  # ldur
    assert enc(ldrh, w1, typed(x0 - 256, ctypes.c_int16)) == enc(ldrh, w1, mem[x0 - 256])


def test_attribute_names_shadowed_by_view_properties():
    class Buf(ctypes.Structure):
        _fields_ = [("addr", ctypes.c_void_p), ("size", ctypes.c_uint32), ("ctype", ctypes.c_uint8)]

    b = typed(x0, Buf)
    assert b.size == ctypes.sizeof(Buf)
    assert b["addr"] == M(x0, 0, 8)
    assert b["size"] == M(x0, 8, 4)
    assert b["ctype"] == M(x0, 12, 1)


def test_underscore_fields_and_dir():
    class Padded(ctypes.Structure):
        _fields_ = [("_pad", ctypes.c_uint32), ("addr", ctypes.c_void_p), ("_n", ctypes.c_int16)]

    v = typed(x0, Padded)
    assert v._pad == M(x0, 0, 4)
    assert v._n == M(x0, 16, 2)
    assert v["addr"] == M(x0, 8, 8)
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

    v = typed(x0, Odd)
    assert v._fields == M(x0, 8, 8)
    assert v["_fields"] == M(x0, 8, 8)
    assert "_fields" in dir(v)


def test_mem_keeps_the_size():
    p = typed(x0, Point)
    x = p.x
    assert mem[x] is x and mem[x].size == 4 and mem[x].extent == 4
    assert enc(ldr, w1, mem[p.y]) == enc(ldr, w1, mem[x0 + 4])
    with pytest.raises(EncodeError, match="holds 4 bytes, this access is 8"):
        enc(ldr, x1, mem[p.x])
    with pytest.raises(EncodeError, match="mem"):
        mem.pre[p.x]


def test_pair_stays_inside_one_array():
    p = typed(x0, Point)
    ints = typed(x1, ctypes.c_int32 * 4)
    # Two consecutive elements of one array.
    assert enc(ldp, w2, w3, ints[0]) == enc(ldp, w2, w3, mem[x1])
    assert enc(stp, w2, w3, ints[2]) == enc(stp, w2, w3, mem[x1 + 8])
    assert enc(ldp, d0, d1, p.v[2]) == enc(ldp, d0, d1, mem[x0 + 40])
    assert enc(ldpsw, x2, x3, ints[1]) == enc(ldpsw, x2, x3, mem[x1 + 4])
    # An element of a nested array has the rest of its own row.
    o = typed(x0, Outer)
    assert enc(ldp, w1, w2, o.grid[1][1]) == enc(ldp, w1, w2, mem[x0 + Outer.grid.offset + 16])
    # The elements of a zero length array are unbounded.
    assert enc(ldp, x2, x3, typed(x1, ctypes.c_int64 * 0)[7]) == enc(ldp, x2, x3, mem[x1 + 56])
    for fn, ops in [
        (ldp, (w1, w2, p.x)),  # x and y are two fields, not an array
        (ldp, (w1, w2, o.pts[0].x)),  # a field of an array element is a field
        (stp, (d0, d1, p.v[3])),  # the last element
        (ldp, (x1, x2, typed(x0, ctypes.c_int64))),
        (ldp, (w1, w2, ints[3])),
        (ldpsw, (x2, x3, ints[3])),
        (ldp, (w1, w2, o.grid[0][2])),  # the inner array ends, although grid[1] follows
        (ldp, (x1, x2, o.num.i)),
    ]:
        with pytest.raises(EncodeError, match="a pair must be two elements of one array"):
            enc(fn, *ops)
    with pytest.raises(EncodeError, match=r"reads 16 bytes from typed\(\) operand \[x0, #48\], which has 8 bytes"):
        enc(ldp, d0, d1, p.v[3])
    # The extent is a bound, not part of the address.
    assert ints[3] == M(x1, 12, 4) and str(ints[3]) == "[x1, #12]"
    assert hash(ints[3]) == hash(M(x1, 12, 4))
    assert ints[0].extent == 16 and ints[3].extent == 4 and p.y.extent == 4
    assert typed(x1, ctypes.c_int64 * 0)[3].extent is None
    with pytest.raises(EncodeError, match="extent"):
        MemExpr(x0, size=8, extent=4)
    with pytest.raises(EncodeError, match="extent"):
        MemExpr(x0, extent=4)
    with pytest.raises(EncodeError, match="size"):
        MemExpr(x0, size=True)
    # Two fields pair through the unsized address of their structure.
    assert enc(ldp, d0, d1, p.addr) == enc(ldp, d0, d1, mem[x0])


def test_typed_is_the_function():
    assert isinstance(A.typed, types.FunctionType)
    from jita.aarch64 import typed as t

    assert t is A.typed
    assert "typed" in A.__all__


def test_errors():
    with pytest.raises(EncodeError, match="bit field"):
        typed(x0, Bits).f
    assert typed(x0, Bits).n == M(x0, 0, 4)

    class Long(ctypes.Structure):
        _fields_ = [("ld", ctypes.c_longdouble)]

    if ctypes.c_longdouble is ctypes.c_double:  # macOS arm64, Windows
        assert typed(x0, Long).ld == M(x0, 0, 8)
    else:
        with pytest.raises(EncodeError, match="long double"):
            typed(x0, Long).ld
    with pytest.raises(AttributeError, match="fields: x, y, tag, next, v"):
        typed(x0, Point).z
    with pytest.raises(AttributeError):
        typed(x0, Point)["_fields_"]
    # Register index: a 1/2/4/8 byte element, no offset and no other index.
    with pytest.raises(EncodeError, match="scale"):
        typed(x0, Point * 2)[x1]
    with pytest.raises(EncodeError, match="both an index register and an offset"):
        typed(x0, Point).v[x1]  # v is at offset 24
    with pytest.raises(EncodeError, match="both an index register and an offset"):
        typed(x0 + 8, ctypes.c_int64 * 4)[x1]
    with pytest.raises(EncodeError, match="already has an index"):
        typed(x0 + (x2 << 1), ctypes.c_int16 * 4)[x1]
    with pytest.raises(EncodeError, match="already has an index"):
        typed(x0, (ctypes.c_int32 * 2) * 4)[x1][x2]
    # Offsets after a register index cannot be encoded.
    with pytest.raises(EncodeError, match="both an index register and an offset"):
        typed(x0, (ctypes.c_int32 * 2) * 4)[x1][1]
    with pytest.raises(EncodeError, match="both an index register and an offset"):
        typed(x0, Num * 2)[x1].b[1]
    # The shift is the element size, and the access must match it.
    with pytest.raises(EncodeError, match="scaled by 8, but a 4 byte access"):
        typed(x0, (ctypes.c_int32 * 2) * 4)[x1][0]
    with pytest.raises(EncodeError, match="scaled by 4, but a 2 byte access"):
        typed(x0 + (x1 << 2), ctypes.c_int16)
    with pytest.raises(EncodeError, match="needs an extend"):
        typed(x0, ctypes.c_int32 * 4)[w1]
    with pytest.raises(EncodeError, match=r"write x1 and let"):
        typed(x0, ctypes.c_int32 * 4)[x1 << 3]
    with pytest.raises(EncodeError, match=r"write w1.uxtw\(\) and let"):
        typed(x0, ctypes.c_int32 * 4)[w1.uxtw(3)]
    with pytest.raises(EncodeError, match=r"write x1.sxtx\(\) and let"):
        typed(x0, ctypes.c_int32 * 4)[x1.sxtx(1)]
    # A shift or extend the index cannot take names the ones it can,
    # whatever the amount, and each suggestion is a working index.
    ints = typed(x0, ctypes.c_int32 * 4)
    for bad in [x1 >> 3, x1.lsr(2), x1.asr(2), x1.uxtw(), x1.uxtx(2), x1.sxtw(2)]:
        with pytest.raises(EncodeError, match=r"a 64 bit index cannot take \w+; write x1 or x1.sxtx\(\)$"):
            ints[bad]
    for wbad in [w1 << 3, w1 << 2, w1.lsr(2), w1.sxtx(), w1.uxtb(2), w1.sxth()]:
        with pytest.raises(EncodeError, match=r"a 32 bit index cannot take \w+; write w1.uxtw\(\) or w1.sxtw\(\)$"):
            ints[wbad]
    with pytest.raises(EncodeError, match=r"needs an extend, w1.uxtw\(\) or w1.sxtw\(\)$"):
        ints[w1]
    assert ints[x1] == M(x0, 0, 4, x1, Mod("lsl", 2))
    assert ints[x1.sxtx()] == M(x0, 0, 4, x1, Mod("sxtx", 2))
    assert ints[w1.uxtw()] == M(x0, 0, 4, w1, Mod("uxtw", 2))
    assert ints[w1.sxtw()] == M(x0, 0, 4, w1, Mod("sxtw", 2))
    with pytest.raises(EncodeError, match="sp cannot be an index register"):
        ints[sp]
    with pytest.raises(EncodeError, match="d1 cannot be an index register"):
        ints[d1]
    # A field narrower than its register indexed element.
    with pytest.raises(EncodeError, match="add the scaled index to the base register first"):
        typed(x0, Num * 4)[x1].b[0]
    with pytest.raises(IndexError):
        typed(x0, Point).v[4]
    with pytest.raises(IndexError):
        typed(x0, Point).v[-1]
    with pytest.raises(TypeError):
        typed(x0, Point).v["x"]
    # Base and type checks.
    with pytest.raises(EncodeError, match="unsized"):
        typed(typed(x0, Point).x, Point)
    with pytest.raises(EncodeError, match="pre-index"):
        typed(mem.pre[x0 + 16], Point)
    with pytest.raises(EncodeError, match="post-index"):
        typed(mem.post[x0, 16], Point)
    with pytest.raises(EncodeError, match="memory base"):
        typed(w0, Point)
    with pytest.raises(EncodeError, match="memory base"):
        typed(xzr, Point)
    with pytest.raises(TypeError):
        typed(0x1000, Point)  # type: ignore[call-overload]  # pyright: ignore[reportCallIssue, reportArgumentType]
    with pytest.raises(TypeError):
        typed(x0, int)
    with pytest.raises(TypeError):
        typed(x0, Point()).x  # type: ignore[call-overload]  # pyright: ignore[reportCallIssue, reportArgumentType]
    with pytest.raises(AttributeError):
        typed(x0, Point).x = 1
    with pytest.raises(EncodeError, match="size"):
        MemExpr(x0, size=3)

    class Big(ctypes.BigEndianStructure):
        _fields_ = [("n", ctypes.c_int32)]

    with pytest.raises(EncodeError, match="big endian"):
        typed(x0, Big).n


def test_flexible_array_member_is_not_bounds_checked():
    class Vec(ctypes.Structure):
        _fields_ = [("n", ctypes.c_int64), ("items", ctypes.c_int64 * 0)]

    assert typed(x0, Vec).items[5] == M(x0, 48, 8)


def build_routine(a):
    """int64 f(Point *p, int64 *vals, uint32 i):
    p->y = p->x + p->y; p->v[p->tag] *= 2; p->next = &p->v[3];
    vals[i] = -vals[i]; return p->x (sign extended).
    """
    p = typed(x0, Point)
    vals = typed(x1, ctypes.c_int64 * 0)
    with a:
        ldr(w3, p.x)
        ldr(w4, p.y)
        add(w4, w3, w4)
        str_(w4, p.y)
        ldrb(w5, p.tag)
        add(x6, x0, Point.v.offset)
        v = typed(x6, ctypes.c_double * 4)
        ldr(d0, v[x5])
        fadd(d0, d0, d0)
        str_(d0, v[x5])
        add(x7, x6, 3 * 8)
        str_(x7, p.next)
        ldr(x8, vals[w2.uxtw()])
        neg(x8, x8)
        str_(x8, vals[w2.uxtw()])
        ldrsw(x0, p.x)
        ret()


ROUTINE_AS = """
ldr w3, [x0]
ldr w4, [x0, #4]
add w4, w3, w4
str w4, [x0, #4]
ldrb w5, [x0, #8]
add x6, x0, #24
ldr d0, [x6, x5, lsl #3]
fadd d0, d0, d0
str d0, [x6, x5, lsl #3]
add x7, x6, #24
str x7, [x0, #16]
ldr x8, [x1, w2, uxtw #3]
neg x8, x8
str x8, [x1, w2, uxtw #3]
ldrsw x0, [x0]
ret
"""


def build_indexes(a):
    b8, s8 = typed(x0, ctypes.c_uint8 * 8), typed(x0, ctypes.c_int8 * 8)
    h, sh = typed(x0, ctypes.c_uint16 * 8), typed(x0, ctypes.c_int16 * 8)
    w, q = typed(x0, ctypes.c_int32 * 8), typed(x0, ctypes.c_int64 * 8)
    f, d = typed(x0, ctypes.c_float * 8), typed(x0, ctypes.c_double * 8)
    with a:
        ldrb(w1, b8[x2])
        ldrb(w1, b8[xzr])
        ldrb(w1, b8[x2.sxtx()])
        ldrb(w1, b8[w2.uxtw()])
        ldrsb(x1, s8[w2.sxtw()])
        ldrh(w1, h[x2])
        strh(w1, h[w2.uxtw()])
        ldrsh(w1, sh[x2.sxtx()])
        ldrsh(x1, sh[w2.sxtw()])
        ldr(w1, w[w2.sxtw()])
        ldr(w1, w[x2.sxtx(2)])
        ldr(x1, q[x2.sxtx()])
        ldr(x1, q[xzr])
        ldr(s0, f[x2])
        str_(s0, f[w2.uxtw()])
        ldr(s1, f[x2.sxtx()])
        ldr(d0, d[xzr])
        str_(d1, d[x2.sxtx()])
        ldr(d2, d[w2.sxtw()])
        ldp(s0, s1, f[2])
        stp(d0, d1, d[6])


INDEXES_AS = """
ldrb w1, [x0, x2]
ldrb w1, [x0, xzr]
ldrb w1, [x0, x2, sxtx]
ldrb w1, [x0, w2, uxtw]
ldrsb x1, [x0, w2, sxtw]
ldrh w1, [x0, x2, lsl #1]
strh w1, [x0, w2, uxtw #1]
ldrsh w1, [x0, x2, sxtx #1]
ldrsh x1, [x0, w2, sxtw #1]
ldr w1, [x0, w2, sxtw #2]
ldr w1, [x0, x2, sxtx #2]
ldr x1, [x0, x2, sxtx #3]
ldr x1, [x0, xzr, lsl #3]
ldr s0, [x0, x2, lsl #2]
str s0, [x0, w2, uxtw #2]
ldr s1, [x0, x2, sxtx #2]
ldr d0, [x0, xzr, lsl #3]
str d1, [x0, x2, sxtx #3]
ldr d2, [x0, w2, sxtw #3]
ldp s0, s1, [x0, #8]
stp d0, d1, [x0, #48]
"""


@requires_aarch64_oracle
def test_register_indexes_match_gnu_as():
    a = Assembler(A)
    build_indexes(a)
    assert bytes(a.cur.buf) == assemble(INDEXES_AS, arch="aarch64")


@requires_aarch64_oracle
def test_routine_bytes_match_gnu_as():
    a = Assembler(A)
    build_routine(a)
    assert bytes(a.cur.buf) == assemble(ROUTINE_AS, arch="aarch64")


@needs_aarch64_host
def test_execute_reads_and_writes_structure():
    a = Assembler(A)
    build_routine(a)
    pt = Point(x=-5, y=12, tag=2)
    pt.v[:] = [1.0, 2.0, 3.5, 4.0]
    vals = (ctypes.c_int64 * 4)(10, 20, 30, 40)
    with a.load() as mod:
        fn = mod.function(ctypes.c_int64, ctypes.POINTER(Point), ctypes.POINTER(ctypes.c_int64), ctypes.c_uint32)
        assert fn(ctypes.byref(pt), vals, 2) == -5
        assert pt.y == 7
        assert list(pt.v) == [1.0, 2.0, 7.0, 4.0]
        assert pt.next == ctypes.addressof(pt) + Point.v.offset + 3 * 8
        assert list(vals) == [10, 20, -30, 40]
