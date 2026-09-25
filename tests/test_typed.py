"""typed(): ctypes structures as views producing sized memory operands."""

import ctypes
import platform
import types

import pytest
from oracle import assemble, requires_oracle

import jita.x64 as x64
from jita import Assembler, EncodeError
from jita.x64 import *  # noqa: F403
from jita.x64.structs import Typed, TypedArray

del test  # noqa: F821, the x64 mnemonic, not a pytest test

needs_host = pytest.mark.skipif(
    platform.machine().lower() not in ("x86_64", "amd64") or platform.system() == "Windows",
    reason="needs an x86-64 POSIX host",
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


def test_spec_example():
    p = typed(rdi, Point)
    assert p.x == dword[rdi]
    assert p.y == dword[rdi + 4]
    assert p.tag == byte[rdi + 8]
    assert p.next == qword[rdi + 16]
    assert p.v[1] == qword[rdi + 32]
    assert p.v[rcx] == qword[rdi + rcx * 8 + 24]
    assert p.v.addr == ptr[rdi + 24]
    assert p.v.addr.size is None
    assert typed(rsi, Point).next == qword[rsi + 16]
    assert p.size == ctypes.sizeof(Point) and p.ctype is Point
    assert p.addr == ptr[rdi]


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

    assert typed(rbx, S).f == MemExpr(base=rbx, disp=8, size=size)
    assert typed(rbx, ctype) == MemExpr(base=rbx, size=size)


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
    base = typed(r12 + 100, ctype)
    leaves = list(_paths(ctype))
    assert leaves
    for path, off, size in leaves:
        assert _walk(base, path) == MemExpr(base=r12, disp=100 + off, size=size), path


def test_layout_details():
    assert typed(rax, Packed).b == qword[rax + 1]
    assert typed(rax, Packed).c == word[rax + 9]
    o = typed(rax, Outer)
    # Anonymous members are reachable directly, at ctypes' offsets.
    assert o.lo == word[rax + Outer.anon.offset]
    assert o.hi == qword[rax + Outer.anon.offset + 8]
    assert isinstance(o.anon, Typed)
    # Union members share an offset.
    assert o.num.i.disp == o.num.f.disp == o.num.b[0].disp == Outer.num.offset
    assert o.num.b[7] == byte[rax + Outer.num.offset + 7]
    # Nested arrays and arrays of structures.
    assert isinstance(o.grid, TypedArray) and isinstance(o.grid[1], TypedArray)
    assert len(o.grid) == 2 and len(o.grid[0]) == 3
    assert o.grid[1][2] == dword[rax + Outer.grid.offset + 12 + 8]
    assert o.pts[2].v[3] == qword[rax + Outer.pts.offset + 2 * 56 + 24 + 24]
    with pytest.raises(EncodeError, match="scale"):
        o.grid[rcx]  # 12 byte rows
    # Pointer fields are plain qwords, never dereferenced.
    assert o.ptr == qword[rax + Outer.ptr.offset]
    assert o.name == qword[rax + Outer.name.offset]
    # Derived structures see the base fields.
    assert typed(rax, Derived).extra == word[rax + ctypes.sizeof(Point)]
    assert typed(rax, Derived).v[0] == qword[rax + 24]


def test_register_index():
    p = typed(rdi + 8, Point)
    assert p.v[r9] == qword[rdi + r9 * 8 + 32]
    assert p.tag.size == 1
    arr = typed(rsi, ctypes.c_int16 * 10)
    assert arr[rdx] == word[rsi + rdx * 2]
    assert typed(rsi, ctypes.c_uint8 * 4)[rdx] == byte[rsi + rdx]
    assert typed(ptr[0x1000], ctypes.c_int64 * 4)[rcx] == qword[rcx * 8 + 0x1000]
    # An index register into an array of 8 byte structures.
    pairs = typed(rdi, (ctypes.c_int32 * 2) * 4)
    assert pairs[rcx][1] == dword[rdi + rcx * 8 + 4]


def test_rip_relative_base():
    t = typed(rip + "table", Point)
    assert t.y == dword[rip + "table" + 4]
    with pytest.raises(EncodeError):
        t.v[rcx]


def test_attribute_names_shadowed_by_view_properties():
    class Buf(ctypes.Structure):
        _fields_ = [("addr", ctypes.c_void_p), ("size", ctypes.c_uint32), ("ctype", ctypes.c_uint8)]

    b = typed(rax, Buf)
    assert b.size == ctypes.sizeof(Buf)
    assert b["addr"] == qword[rax]
    assert b["size"] == dword[rax + 8]
    assert b["ctype"] == byte[rax + 12]


def test_underscore_fields_and_dir():
    class Padded(ctypes.Structure):
        _fields_ = [("_pad", ctypes.c_uint32), ("addr", ctypes.c_void_p), ("_n", ctypes.c_int16)]

    v = typed(rax, Padded)
    assert v._pad == dword[rax]
    assert v._n == word[rax + 16]
    assert v["addr"] == qword[rax + 8]
    names = dir(v)
    assert len(names) == len(set(names))
    assert set(names) == {"_pad", "_n", "addr", "ctype", "size"}
    with pytest.raises(AttributeError, match="fields: _pad, addr, _n"):
        v._missing
    with pytest.raises(AttributeError):
        v.__missing__


def test_typed_is_the_function():
    import jita.x64.structs  # noqa: F401

    assert isinstance(x64.typed, types.FunctionType)
    from jita.x64 import typed as t

    assert t is x64.typed


def test_errors():
    with pytest.raises(EncodeError, match="bit field"):
        typed(rax, Bits).f
    assert typed(rax, Bits).n == dword[rax]

    class Long(ctypes.Structure):
        _fields_ = [("ld", ctypes.c_longdouble)]

    with pytest.raises(EncodeError, match="long double"):
        typed(rax, Long).ld
    with pytest.raises(AttributeError, match="fields: x, y, tag, next, v"):
        typed(rax, Point).z
    with pytest.raises(AttributeError):
        typed(rax, Point)["_fields_"]
    # Register index needs a 1/2/4/8 byte element and a free index slot.
    with pytest.raises(EncodeError, match="scale"):
        typed(rax, Point * 2)[rcx]
    with pytest.raises(EncodeError, match="already has an index"):
        typed(rax + rbx * 2, Point).v[rcx]
    with pytest.raises(EncodeError, match="already has an index"):
        typed(rax, (ctypes.c_int32 * 2) * 4)[rcx][rdx]
    with pytest.raises(IndexError):
        typed(rax, Point).v[4]
    with pytest.raises(IndexError):
        typed(rax, Point).v[-1]
    with pytest.raises(TypeError):
        typed(rax, Point).v["x"]
    # Base and type checks.
    with pytest.raises(EncodeError, match="unsized"):
        typed(qword[rax], Point)
    with pytest.raises(TypeError):
        typed(0x1000, Point)  # type: ignore[call-overload]  # pyright: ignore[reportCallIssue, reportArgumentType]
    with pytest.raises(TypeError):
        typed(rax, int)
    with pytest.raises(TypeError):
        typed(rax, Point()).x  # type: ignore[call-overload]  # pyright: ignore[reportCallIssue, reportArgumentType]
    with pytest.raises(AttributeError):
        typed(rax, Point).x = 1

    class Big(ctypes.BigEndianStructure):
        _fields_ = [("n", ctypes.c_int32)]

    with pytest.raises(EncodeError, match="big endian"):
        typed(rax, Big).n


def test_flexible_array_member_is_not_bounds_checked():
    class Vec(ctypes.Structure):
        _fields_ = [("n", ctypes.c_int64), ("items", ctypes.c_int64 * 0)]

    assert typed(rdi, Vec).items[5] == qword[rdi + 48]


def build_routine(a):
    """int64 f(Point *p): p->y = p->x + p->y; p->v[p->tag] *= 2; return p->x.

    Also stores the element address of v[3] into p->next.
    """
    p = typed(rdi, Point)
    with a:
        mov(eax, p.x)
        add(eax, p.y)
        mov(p.y, eax)
        movzx(ecx, p.tag)
        movsd(xmm0, p.v[rcx])
        addsd(xmm0, xmm0)
        movsd(p.v[rcx], xmm0)
        lea(rdx, p.v[3])
        mov(p.next, rdx)
        movsxd(rax, p.x)
        ret()


ROUTINE_AS = """
mov eax, dword ptr [rdi]
add eax, dword ptr [rdi+4]
mov dword ptr [rdi+4], eax
movzx ecx, byte ptr [rdi+8]
movsd xmm0, qword ptr [rdi+rcx*8+24]
addsd xmm0, xmm0
movsd qword ptr [rdi+rcx*8+24], xmm0
lea rdx, [rdi+48]
mov qword ptr [rdi+16], rdx
movsxd rax, dword ptr [rdi]
ret
"""


@requires_oracle
def test_routine_bytes_match_gnu_as():
    a = Assembler(x64)
    build_routine(a)
    assert bytes(a.cur.buf) == assemble(ROUTINE_AS)


@needs_host
def test_execute_reads_and_writes_structure():
    a = Assembler(x64)
    build_routine(a)
    pt = Point(x=-5, y=12, tag=2)
    pt.v[:] = [1.0, 2.0, 3.5, 4.0]
    with a.load() as mod:
        fn = mod.function(ctypes.c_int64, ctypes.POINTER(Point))
        assert fn(ctypes.byref(pt)) == -5
        assert pt.y == 7
        assert list(pt.v) == [1.0, 2.0, 7.0, 4.0]
        assert pt.next == ctypes.addressof(pt) + Point.v.offset + 3 * 8

