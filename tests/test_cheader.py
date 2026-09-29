"""jita.cheader: C headers loaded as ctypes types, macros and prototypes."""

import ctypes
import platform
import sys
import types
from _ctypes import CFuncPtr
from pathlib import Path

import pytest

import jita.aarch64 as A
import jita.loongarch64 as L
import jita.riscv64 as R
import jita.x64 as X
from jita import Assembler
from jita.cheader import Diagnostic, Header, HeaderError, diagnostics, load

FIXTURES = Path(__file__).resolve().parent / "cheader"


@pytest.fixture(autouse=True)
def _dependencies(request):
    if request.node.name != "test_load_without_dependencies":
        pytest.importorskip("pycparser")
        pytest.importorskip("pcpp")


def load_text(tmp_path: Path, text: str, name: str = "t.h", **kw) -> Header:
    path = tmp_path / name
    path.write_text(text)
    return load(path, **kw)


def skips(h: Header) -> dict[str | None, Diagnostic]:
    return {d.name: d for d in diagnostics(h) if d.severity == "skip"}


def notes(h: Header, kind: str) -> list[Diagnostic]:
    return [d for d in diagnostics(h) if d.severity == "note" and d.kind == kind]


def offsets(ctype: type) -> dict[str, int]:
    return {f[0]: getattr(ctype, f[0]).offset for f in getattr(ctype, "_fields_")}


@pytest.fixture(scope="module")
def types_h() -> Header:
    pytest.importorskip("pycparser")
    pytest.importorskip("pcpp")
    return load(FIXTURES / "types.h")


@pytest.fixture(scope="module")
def review_h() -> Header:
    pytest.importorskip("pycparser")
    pytest.importorskip("pcpp")
    return load(FIXTURES / "review.h")


# -- the module ----------------------------------------------------------------


def test_load_without_dependencies(monkeypatch):
    monkeypatch.setitem(sys.modules, "pycparser", None)
    with pytest.raises(ImportError, match=r"jita\[cheader\]"):
        load(FIXTURES / "types.h")


def test_header_is_a_module(types_h):
    h = types_h
    assert isinstance(h, Header) and isinstance(h, types.ModuleType)
    assert h.__name__ == "jita.cheader.types"
    assert h.__file__ == str(FIXTURES / "types.h")
    assert "names from" in (h.__doc__ or "")
    assert "struct_Vec" in dir(h) and "Vec" in vars(h)
    assert h.struct_Vec.__module__ == h.__name__
    assert sys.modules.get(h.__name__) is None
    with pytest.raises(AttributeError, match="has no attribute 'nothing'"):
        getattr(h, "nothing")


def test_every_load_builds_new_classes():
    a = load(FIXTURES / "types.h")
    b = load(FIXTURES / "types.h")
    assert a is not b and a.struct_Vec is not b.struct_Vec


def test_header_lookup(tmp_path):
    (tmp_path / "inc").mkdir()
    (tmp_path / "inc" / "a.h").write_text("struct A { int x; };\n")
    assert ctypes.sizeof(load("a.h", include_dirs=[tmp_path / "inc"]).struct_A) == 4
    with pytest.raises(HeaderError, match="cannot find header 'missing.h'"):
        load("missing.h", include_dirs=[tmp_path])
    with pytest.raises(TypeError):
        load()


# -- 1. scalars ---------------------------------------------------------------


def test_scalars(types_h):
    t = {f[0]: f[1] for f in types_h.struct_Scalars._fields_}
    c = ctypes
    assert (t["c"], t["sc"], t["uc"]) == (c.c_char, c.c_byte, c.c_ubyte)
    assert (t["s"], t["us"], t["i"], t["u"], t["ui"]) == (c.c_short, c.c_ushort, c.c_int, c.c_uint, c.c_uint)
    assert (t["l"], t["ul"], t["ll"], t["ull"]) == (c.c_long, c.c_ulong, c.c_longlong, c.c_ulonglong)
    assert (t["f"], t["d"], t["ld"]) == (c.c_float, c.c_double, c.c_longdouble)
    assert (t["b"], t["b2"]) == (c.c_bool, c.c_bool)
    assert (t["i8"], t["u8"], t["i16"], t["u16"]) == (c.c_int8, c.c_uint8, c.c_int16, c.c_uint16)
    assert (t["i32"], t["u32"], t["i64"], t["u64"]) == (c.c_int32, c.c_uint32, c.c_int64, c.c_uint64)
    assert (t["sz"], t["pd"], t["ip"], t["up"], t["wc"]) == (
        c.c_size_t,
        c.c_ssize_t,
        c.c_ssize_t,
        c.c_size_t,
        c.c_wchar,
    )
    assert t["cv"] is c.c_int


def test_complex(tmp_path):
    h = load_text(tmp_path, "struct C { float _Complex f; double _Complex d; };\n")
    if hasattr(ctypes, "c_double_complex"):
        fields = dict(h.struct_C._fields_)
        assert (fields["f"], fields["d"]) == (ctypes.c_float_complex, ctypes.c_double_complex)
    else:
        assert "struct_C" in skips(h)


# -- 2. typedefs ---------------------------------------------------------------


def test_typedef_chains(types_h):
    h = types_h
    assert h.my_ssize is ctypes.c_long and h.my_hash2 is ctypes.c_long
    assert h.Vec is h.struct_Vec and h.Vec.__name__ == "struct_Vec"
    assert h.Kind.__name__ == "Kind" and ctypes.POINTER(h.Kind).__name__ == "LP_Kind"
    assert h.PKind is ctypes.POINTER(h.Kind)
    assert "struct_Kind" not in vars(h)


def test_anonymous_struct_takes_a_later_typedef_name(tmp_path):
    h = load_text(tmp_path, "typedef struct { int a; } *PT, T;\n")
    assert h.T.__name__ == "T" and h.PT is ctypes.POINTER(h.T)


def test_std_names_win_over_the_header(tmp_path):
    h = load_text(tmp_path, "typedef unsigned int uint64_t;\ntypedef long Py_ssize_t;\nstruct S { uint64_t x; };\n")
    assert dict(h.struct_S._fields_)["x"] is ctypes.c_uint64
    assert h.Py_ssize_t is ctypes.c_long


# -- 3. pointers and arrays ----------------------------------------------------


def test_pointers(types_h):
    h = types_h
    t = dict(h.struct_Pointers._fields_)
    assert t["v"] is ctypes.POINTER(h.struct_Vec)
    assert (t["s"], t["cs"], t["p"], t["ws"]) == (ctypes.c_char_p, ctypes.c_char_p, ctypes.c_void_p, ctypes.c_wchar_p)
    assert t["pp"] is ctypes.POINTER(ctypes.c_void_p)
    assert t["arr"] is ctypes.POINTER(ctypes.c_int * 4)
    assert t["bytes"] is ctypes.POINTER(ctypes.c_ubyte)


def test_arrays(types_h):
    h = types_h
    t = dict(h.struct_Arrays._fields_)
    assert t["a"]._length_ == 8 and t["a"]._type_ is ctypes.c_int
    assert t["m"]._length_ == 3 and t["m"]._type_._length_ == 4
    assert ctypes.sizeof(t["m"]) == 96 and ctypes.sizeof(t["vs"]) == 2 * ctypes.sizeof(h.Vec)
    flex = dict(h.struct_Flex._fields_)["data"]
    assert flex._length_ == 0 and ctypes.sizeof(h.struct_Flex) == ctypes.sizeof(ctypes.c_size_t)


# -- 4. self and mutual references ----------------------------------------------


def test_references(types_h):
    h = types_h
    assert dict(h.struct_Node._fields_)["next"] is ctypes.POINTER(h.struct_Node)
    assert ctypes.sizeof(h.struct_B) == ctypes.sizeof(h.struct_A) + 8
    assert dict(h.struct_A._fields_)["b"] is ctypes.POINTER(h.struct_B)
    assert dict(h.struct_UsesLater._fields_)["p"] is ctypes.POINTER(h.struct_Later)
    assert ctypes.sizeof(h.struct_Later) == 2


# -- 5. opaque types -------------------------------------------------------------


def test_opaque_types(tmp_path):
    h = load_text(
        tmp_path,
        """
        struct X;
        typedef struct X X_t;
        struct ByPtr { struct X *x; X_t *y; };
        struct ByValue { struct X x; };
        struct ByArray { struct X xs[4]; };
        struct ByTypedef { X_t x; };
        """,
    )
    assert not hasattr(h.struct_X, "_fields_") and h.X_t is h.struct_X
    assert ctypes.sizeof(h.struct_ByPtr) == 16
    s = skips(h)
    for name in ("struct_ByValue", "struct_ByArray", "struct_ByTypedef"):
        assert "opaque" in s[name].message and s[name].kind == "unsupported"
    with pytest.raises(AttributeError, match=r"struct_ByValue was skipped: field x: struct_X is opaque .*t\.h:5\)"):
        getattr(h, "struct_ByValue")
    with pytest.raises(AttributeError, match=r"struct_X is opaque \(only forward declared\)"):
        X.typed(X.rdi, h.struct_X).anything


def test_definition_lost_to_a_parse_error(tmp_path):
    h = load_text(
        tmp_path,
        """
        struct D { typeof(int) x; };
        struct UsesD { struct D d; };
        struct PtrD { struct D *d; };
        """,
    )
    s = skips(h)
    assert s[None].kind == "parse-error" and s[None].line == 2
    assert s["struct_UsesD"].message == "field d: struct_D was skipped (its definition could not be parsed)"
    assert ctypes.sizeof(h.struct_PtrD) == 8


# -- 6. nesting ------------------------------------------------------------------


def test_nested_and_anonymous(types_h):
    h = types_h
    n = h.struct_Nested
    assert n._anonymous_ == ("_jita_anon1", "_jita_anon2")
    assert dict(n._fields_)["inner"] is h.struct_Inner
    assert (n.i.offset, n.f.offset) == (24, 24) and (n.c2.offset, n.s2.offset) == (28, 30)
    assert n._jita_anon0.offset == 32
    p = X.typed(X.rdi, n)
    assert p.i.disp == 24 and p.s2.disp == 30 and p.inner.l.disp == 16
    assert not hasattr(h, "anon_struct1")


# -- 7. enums --------------------------------------------------------------------


def test_enums(types_h):
    h = types_h
    assert (h.RED, h.GREEN, h.BLUE, h.NEG, h.EXPR, h.PREV) == (0, 5, 6, -3, 16, 7)
    assert h.enum_Color.BLUE == 6 and issubclass(h.enum_Color, ctypes.c_int)
    assert h.KindE.__name__ == "KindE" and (h.K_A, h.K_B) == (-1, 0)
    assert issubclass(h.enum_Wide, ctypes.c_uint) and h.W_BIG == 0x80000000
    assert issubclass(h.enum_Huge, ctypes.c_uint64) and issubclass(h.enum_HugeNeg, ctypes.c_int64)
    s = h.struct_WithEnum(color=h.BLUE)
    assert isinstance(s.color, h.enum_Color) and s.color.value == 6


def test_enum_member_clashing_with_ctypes(review_h):
    h = review_h
    assert h.value == 1 and h.enum_E.RED == 2
    assert "value" not in vars(h.enum_E)
    assert [d.name for d in notes(h, "shadowed-member")] == ["value"]


# -- 8. bit fields ---------------------------------------------------------------


def test_bit_fields(types_h):
    h = types_h
    b = h.struct_Bits
    assert [f[0] for f in b._fields_] == ["a", "b", "c", "_jita_bits0", "d"]
    assert b._fields_[3][2] == 4
    if sys.platform != "win32":
        # GCC's rules; MSVC's differ (and the oracle test checks both).
        assert ctypes.sizeof(b) == 4 and b.d.offset == 2
        assert (ctypes.sizeof(h.struct_Bf2), h.struct_Bf2.e.offset) == (16, 15)
        assert ctypes.sizeof(h.struct_Bf3) == 16
        assert (ctypes.sizeof(h.struct_Bf4), h.struct_Bf4.b.offset) == (8, 4)
    with pytest.raises(Exception, match="bit field"):
        X.typed(X.rdi, b).a


def test_zero_width_bit_field(review_h):
    s = skips(review_h)
    assert "zero width bit field" in s["struct_fbits"].message
    assert ctypes.sizeof(review_h.struct_fbits2) == 4


# -- 9. packed and aligned -------------------------------------------------------


def test_packed_and_aligned(types_h):
    h = types_h
    for name, size, pack in [("Packed1", 5, 1), ("Packed2", 5, 1), ("PackedA16", 17, 1), ("PackedAligned", 6, 1)]:
        t = getattr(h, f"struct_{name}")
        assert (ctypes.sizeof(t), t._pack_, t._layout_) == (size, pack, "ms"), name
    assert h.struct_PackedA16.a.offset == 1
    assert h.struct_PackedAligned._align_ == 2
    assert (ctypes.sizeof(h.struct_Aligned16), h.struct_Aligned16._align_) == (16, 16)
    assert ctypes.sizeof(h.A16) == 16
    assert ctypes.sizeof(h.struct_PackedLd) == 1 + ctypes.sizeof(ctypes.c_longdouble)


def test_pragma_pack(types_h):
    h = types_h
    assert (ctypes.sizeof(h.struct_Pragma1), h.struct_Pragma1.s.offset) == (7, 5)
    assert (ctypes.sizeof(h.struct_Pragma2), h.struct_Pragma2.i.offset) == (6, 2)
    assert ctypes.sizeof(h.struct_AfterPragma) == 8 and not hasattr(h.struct_AfterPragma, "_pack_")


def test_declspec_align(tmp_path):
    h = load_text(
        tmp_path,
        """
        __declspec(align(16)) struct S1 { char c; };
        typedef __declspec(align(32)) struct { char c; } S2;
        struct S3 { char c; } __declspec(align(8));
        struct S4 { char c; __declspec(align(2)) short s; };
        struct S5 { char c; __declspec(align(8)) short s; };
        __declspec(dllimport) int f(void);
        """,
    )
    assert (ctypes.sizeof(h.struct_S1), ctypes.sizeof(h.S2), ctypes.sizeof(h.struct_S3)) == (16, 32, 8)
    assert h.struct_S4.s.offset == 2
    assert "alignment 8 exceeds" in skips(h)["struct_S5"].message
    assert h.f._restype_ is ctypes.c_int


def test_unsupported_packing(review_h, tmp_path):
    s = skips(review_h)
    assert "packed struct" in s["struct_pk"].message
    assert "packed" in s["enum_PE"].message
    assert s["struct_usepe"].kind == "skipped-dependency" and "enum_PE" in s["struct_usepe"].message
    h = load_text(tmp_path, "struct P { char c;\n#pragma pack(1)\n int i; };\nstruct Q { int x; };\n")
    assert "#pragma pack inside" in skips(h)["struct_P"].message
    assert ctypes.sizeof(h.struct_Q) == 4


# -- 10. field alignment ---------------------------------------------------------


def test_field_alignment(types_h, review_h):
    fa = types_h.struct_FieldAlign
    assert offsets(fa) == {"c": 0, "i": 4, "d": 8, "s": 10, "e": 12, "j": 16, "f": 20, "t": 22, "g": 24, "h": 25}
    assert offsets(review_h.struct_al) == {"c": 0, "i": 4} and offsets(review_h.struct_fal) == {"c": 0, "i": 4}
    s = skips(review_h)
    assert "alignment 16 exceeds" in s["struct_al2"].message and "alignment 16 exceeds" in s["struct_fal2"].message


def test_field_attribute_owners(tmp_path):
    h = load_text(
        tmp_path,
        """
        struct Trailing { char a; int b __attribute__((aligned(16))); char c; };
        struct Leading { char a; __attribute__((aligned(16))) char b, c; };
        struct OneLine { char a; char b __attribute__((aligned(16))), c; };
        struct Fine { char a __attribute__((aligned(1))); int b; };
        struct Outer { struct { char x __attribute__((aligned(16))); } in; char y; };
        struct Before { char a; __attribute__((aligned(16))) struct { int p; int q; } s; };
        """,
    )
    s = skips(h)
    assert "field b:" in s["struct_Trailing"].message
    assert "field b:" in s["struct_Leading"].message
    assert "field b:" in s["struct_OneLine"].message
    assert "struct_Fine" not in s
    assert "field in:" in s["struct_Outer"].message
    assert "field s:" in s["struct_Before"].message


def test_unsupported_attribute_on_a_definition(tmp_path):
    h = load_text(tmp_path, "struct S { int x; } __attribute__((vector_size(16)));\nstruct T { struct S s; };\n")
    s = skips(h)
    assert "vector_size" in s["struct_S"].message and "struct_S" not in vars(h)
    assert s["struct_T"].kind == "skipped-dependency"


def test_aligned_typedef(tmp_path):
    h = load_text(tmp_path, "typedef int aint __attribute__((aligned(16)));\nstruct U { aint x; };\n")
    s = skips(h)
    assert s["aint"].kind == "unsupported" and s["struct_U"].kind == "skipped-dependency"


# -- 11. attributes ---------------------------------------------------------------


def test_harmless_attributes(tmp_path):
    h = load_text(
        tmp_path,
        """
        __attribute__((visibility("default"))) int f1(void) __attribute__((nonnull(1), warn_unused_result));
        int f2(const char *fmt) __attribute__((__format__(__printf__, 1, 0), deprecated("x")));
        struct __attribute__((__deprecated__)) S { int x __attribute__((unused)); } __attribute__((may_alias));
        __declspec(noreturn) void f3(void);
        """,
    )
    assert not skips(h)
    assert ctypes.sizeof(h.struct_S) == 4 and h.f1._restype_ is ctypes.c_int


@pytest.mark.parametrize(
    "attr",
    ["mode(QI)", "vector_size(16)", "aligned", "aligned(sizeof(long))", "__vector_size__(8)"],
)
def test_layout_changing_attributes_skip(tmp_path, attr):
    h = load_text(tmp_path, f"typedef int T __attribute__(({attr}));\nstruct U {{ T x; }};\nstruct V {{ int y; }};\n")
    s = skips(h)
    assert s["T"].kind == "unsupported" and "attribute" in s["T"].message
    assert s["struct_U"].kind == "skipped-dependency"
    assert ctypes.sizeof(h.struct_V) == 4


def test_review_attribute_cases(review_h):
    s = skips(review_h)
    assert s["v4si"].kind == "unsupported" and s["struct_vec"].kind == "skipped-dependency"
    assert s["qi_t"].kind == "unsupported" and s["struct_q"].kind == "skipped-dependency"


# -- 12. functions ------------------------------------------------------------------


def test_function_pointer_fields(types_h):
    h = types_h
    t = dict(h.struct_Callbacks._fields_)
    assert h.cb_t._argtypes_ == (ctypes.c_void_p, ctypes.c_int) and h.cb_t._restype_ is ctypes.c_int
    assert t["fn"] is h.cb_t and issubclass(h.cb_t, CFuncPtr)
    assert t["raw"]._argtypes_ == () and t["raw"]._restype_ is None
    assert X.typed(X.rdi, h.struct_Callbacks).two.size == 8
    assert h.struct_Callbacks.tail.offset == 24


def test_prototypes(tmp_path):
    h = load_text(
        tmp_path,
        """
        #include <stdint.h>
        typedef struct Vec { int64_t *data; } Vec;
        int64_t sum(const Vec *v, int k);
        void none(void);
        Vec byval(Vec v);
        int decay(int a[4], int (*cb)(int), int f(double));
        typedef int fn_t(int);
        fn_t *pick(fn_t *f);
        int printf(const char *fmt, ...);
        static inline int twice(int x) { return __builtin_expect(x, 1) ? ({ int y = x; y * 2; }) : 0; }
        """,
    )
    assert h.sum._restype_ is ctypes.c_int64 and h.sum._argtypes_ == (ctypes.POINTER(h.Vec), ctypes.c_int)
    assert h.none._argtypes_ == () and h.none._restype_ is None
    assert h.byval._argtypes_ == (h.Vec,)
    ptr, cb, f = h.decay._argtypes_
    assert ptr is ctypes.POINTER(ctypes.c_int) and cb._argtypes_ == (ctypes.c_int,)
    assert f._argtypes_ == (ctypes.c_double,)
    assert h.fn_t._argtypes_ == (ctypes.c_int,) and h.pick._restype_ is h.fn_t
    assert h.twice._argtypes_ == (ctypes.c_int,)
    assert skips(h)["printf"].message == "variadic function"
    with pytest.raises(AttributeError, match=r"printf was skipped: variadic function \(t\.h:10\)"):
        getattr(h, "printf")


# -- 13. macros ---------------------------------------------------------------------


def test_macros(tmp_path):
    h = load_text(
        tmp_path,
        """
        enum { E1 = 4 };
        #define DEC 42
        #define HEX 0x7f
        #define OCT 0755
        #define CHR 'A'
        #define ESC '\\n'
        #define UL 16UL
        #define BIG (3ULL << 30)
        #define NEG (-5)
        #define PAREN ((1 + 2) * 3)
        #define SHIFT (1 << 10)
        #define INV (~0U)
        #define MOD (-7 % 2)
        #define DIV (-7 / 2)
        #define REF (BIG + 1)
        #define ENUM (E1 | 1)
        #define SMAX (SSIZE_MAX - 1)
        #define STR "vec"
        #define CAT "ab" "cd"
        #define FLT 1.5
        #define FLTF 2.5f
        #define CAST ((unsigned char)-1)
        #define ULCAST ((unsigned long)-1)
        #define CMP (DEC > 40 && HEX != 0)
        #define TERN (DEC ? 1 : 2)
        #define FUNC(x) ((x) * 2)
        #define SIZE (sizeof(int))
        #define UNKNOWN_ID (NOT_DEFINED + 1)
        #define EMPTY
        #define __dunder__ 1
        int declared(void);
        #define declared 3
        """,
    )
    lmax = (1 << (8 * ctypes.sizeof(ctypes.c_ssize_t) - 1)) - 1
    ulong_max = (1 << (8 * ctypes.sizeof(ctypes.c_ulong))) - 1
    expected = {
        "DEC": 42, "HEX": 127, "OCT": 0o755, "CHR": 65, "ESC": 10, "UL": 16, "BIG": 3221225472, "NEG": -5,
        "PAREN": 9, "SHIFT": 1024, "INV": 4294967295, "MOD": -1, "DIV": -3, "REF": 3221225473, "ENUM": 5,
        "SMAX": lmax - 1, "STR": "vec", "CAT": "abcd", "FLT": 1.5, "FLTF": 2.5, "CAST": 255,
        "ULCAST": ulong_max, "CMP": 1, "TERN": 1,
    }  # fmt: skip
    assert {k: getattr(h, k) for k in expected} == expected
    assert all(type(getattr(h, k)) is type(v) for k, v in expected.items())
    for name in ("FUNC", "SIZE", "UNKNOWN_ID", "EMPTY", "__dunder__"):
        assert name not in vars(h)
    assert callable(h.declared) and [d.name for d in notes(h, "shadowed-macro")] == ["declared"]


def test_review_macros(review_h):
    h = review_h
    assert (h.BIG, h.NEG, h.INV, h.FLT, h.STR, h.DIV, h.REF) == (
        3221225472,
        -1,
        4294967295,
        1.5,
        "abcd",
        -3,
        3221225473,
    )
    assert (h.BR, h.HB) == (1, 0)
    assert "UNK" not in vars(h)


# -- 14. GNU and MSVC extensions -------------------------------------------------------


def test_extensions(tmp_path):
    h = load_text(
        tmp_path,
        """
        __extension__ typedef long long ll_t;
        struct R { int *__restrict p; __signed__ char c; volatile int v; };
        static __inline__ int inl(int x) { return x; }
        int renamed(void) __asm__("other_name");
        _Static_assert(sizeof(int) == 4, "int");
        struct SA { int x; _Static_assert(1, "in body"); int y; };
        struct I128 { __int128 w; };
        typeof(int) tv;
        struct After { int z; };
        """,
    )
    s = skips(h)
    assert h.ll_t is ctypes.c_longlong and ctypes.sizeof(h.struct_R) == 16
    assert h.inl._argtypes_ == (ctypes.c_int,) and h.renamed._restype_ is ctypes.c_int
    assert ctypes.sizeof(h.struct_SA) == 8
    assert s["struct_I128"].kind == "unsupported" and "__int128" in s["struct_I128"].message
    assert s[None].kind == "parse-error" and s[None].line == 9
    assert ctypes.sizeof(h.struct_After) == 4


# -- 15. preprocessing -----------------------------------------------------------------


def test_includes(tmp_path):
    (tmp_path / "inc").mkdir()
    (tmp_path / "inc" / "other.h").write_text("#pragma once\nstruct Other { int x; };\n")
    h = load_text(
        tmp_path, '#include <stdio.h>\n#include "other.h"\n#include "other.h"\n', include_dirs=[tmp_path / "inc"]
    )
    assert ctypes.sizeof(h.struct_Other) == 4
    missing = notes(h, "missing-include")
    assert [d.message for d in missing] == ["<stdio.h> is not in include_dirs, ignored"]
    assert missing[0].file == str(tmp_path / "t.h") and missing[0].line == 1
    with pytest.raises(HeaderError, match=r't\.h:1: cannot find "nope\.h"'):
        load_text(tmp_path, '#include "nope.h"\n')


def test_conditionals(tmp_path):
    (tmp_path / "other.h").write_text("")
    h = load_text(
        tmp_path,
        """
        #if UNDEFINED_MACRO
        #define A 1
        #else
        #define A 0
        #endif
        #if 1LL
        #define B 1
        #endif
        #if 0xffffffffffffffffULL > 0
        #define C 1
        #endif
        #if (3ULL << 30) == 3221225472
        #define D 1
        #endif
        #if __has_builtin(__builtin_expect) || __has_attribute(packed) || __has_feature(x)
        #define E 1
        #else
        #define E 0
        #endif
        #if defined(__has_include) && __has_include("other.h") && !__has_include(<nothere.h>)
        #define F 1
        #endif
        #ifdef CHOICE
        #define G CHOICE
        #endif
        #ifdef GONE
        #define H 1
        #endif
        #error something is off
        struct S { int x; };
        """,
        include_dirs=[tmp_path],
        defines={"CHOICE": 7, "GONE": None, "__GNUC__": None},
        strict=True,
    )
    assert (h.A, h.B, h.C, h.D, h.E, h.F, h.G) == (0, 1, 1, 1, 0, 1, 7)
    assert "H" not in vars(h)
    assert [d.message for d in notes(h, "preprocessor")] == ["#error something is off"]


def test_compiler_identity(tmp_path):
    h = load_text(
        tmp_path,
        """
        #ifdef _MSC_VER
        #define MSC 1
        #endif
        #ifdef __GNUC__
        #define GNU 1
        #endif
        #define PTR __SIZEOF_POINTER__
        """,
    )
    if sys.platform == "win32":
        assert h.MSC == 1 and "GNU" not in vars(h)
    else:
        assert h.GNU == 1 and "MSC" not in vars(h) and h.PTR == ctypes.sizeof(ctypes.c_void_p)


# -- 16. system types -----------------------------------------------------------------


def test_unknown_system_types(tmp_path):
    text = "#include <stdio.h>\nstruct F { FILE *fp; };\nstruct O { off_t off; };\n"
    h = load_text(tmp_path, text)
    assert dict(h.struct_F._fields_)["fp"] is ctypes.c_void_p
    assert any(d.name == "fp" and "FILE" in d.message for d in notes(h, "unsupported"))
    s = skips(h)
    assert "types={'off_t'" in s["struct_O"].message and s["struct_O"].kind == "skipped-dependency"
    h = load_text(tmp_path, text, types={"off_t": ctypes.c_int64})
    assert dict(h.struct_O._fields_)["off"] is ctypes.c_int64


# -- 17. dependencies -------------------------------------------------------------------


def test_dependency_cascade(tmp_path):
    h = load_text(
        tmp_path,
        """
        struct A { __int128 w; };
        struct B { struct A a; };
        struct C { struct A *a; };
        typedef struct A A_t;
        struct D { A_t *a; };
        """,
    )
    s = skips(h)
    assert s["struct_A"].kind == "unsupported"
    assert s["struct_B"].kind == "skipped-dependency" and "struct_A" in s["struct_B"].message
    assert ctypes.sizeof(h.struct_C) == 8
    assert s["A_t"].kind == "skipped-dependency"
    assert dict(h.struct_D._fields_)["a"] is ctypes.c_void_p


# -- 18. strict ----------------------------------------------------------------------------


def test_strict(tmp_path):
    with pytest.raises(HeaderError, match=r"t\.h:1: struct_A: field w: __int128"):
        load_text(tmp_path, "struct A { __int128 w; };\n", strict=True)
    with pytest.raises(HeaderError, match=r"t\.h:2: .*: typeof\(int\) y;"):
        load_text(tmp_path, "int x;\ntypeof(int) y;\n", strict=True)
    with pytest.raises(HeaderError, match="variadic"):
        load_text(tmp_path, "int f(int, ...);\n", strict=True)
    h = load_text(tmp_path, "#include <stdio.h>\nstruct F { FILE *f; };\n", strict=True)
    assert ctypes.sizeof(h.struct_F) == 8


# -- 19. filters ------------------------------------------------------------------------------


def test_filters(tmp_path):
    text = "struct Inner { int x; };\nstruct Py_Outer { struct Inner in; };\ntypedef int _private;\n"
    h = load_text(tmp_path, text, names=["*Py_*"], exclude=["_*"])
    assert "struct_Py_Outer" in vars(h) and "struct_Inner" not in vars(h) and "_private" not in vars(h)
    assert dict(h.struct_Py_Outer._fields_)["in"].__name__ == "struct_Inner"
    for prelude_name in ("FILE", "size_t", "INT_MAX", "bool", "SEEK_SET"):
        assert prelude_name not in vars(load_text(tmp_path, text))


# -- 20. several headers --------------------------------------------------------------------


def test_several_headers(tmp_path):
    (tmp_path / "a.h").write_text("typedef struct { int x; } A;\n")
    (tmp_path / "b.h").write_text('#include "a.h"\nstruct B { A a; };\n')
    h = load(tmp_path / "a.h", tmp_path / "b.h")
    assert ctypes.sizeof(h.struct_B) == 4 and h.__name__ == "jita.cheader.a"


def test_header_included_twice(review_h, tmp_path):
    assert ctypes.sizeof(review_h.struct_Twice) == 4
    (tmp_path / "noguard.h").write_text("struct Twice { int a; };\n")
    h = load_text(tmp_path, '#include "noguard.h"\n#include "noguard.h"\n')
    assert ctypes.sizeof(h.struct_Twice) == 4 and not skips(h)


# -- 21. recovery -------------------------------------------------------------------------------


def test_recovery(tmp_path):
    (tmp_path / "bad.h").write_text("typedef typeof(int) T;\nstruct Before { int b; };\n")
    h = load_text(
        tmp_path,
        """
        #include "bad.h"
        struct ByPtr { T *t; };
        struct ByValue { T t; };
        #include "bad.h"
        """,
        defines={"_": 1},
    )
    s = skips(h)
    assert s["T"].kind == "parse-error" and s["T"].file == str(tmp_path / "bad.h") and s["T"].line == 1
    assert dict(h.struct_ByPtr._fields_)["t"] is ctypes.c_void_p
    assert s["struct_ByValue"].kind == "skipped-dependency" and "could not be parsed" in s["struct_ByValue"].message
    assert ctypes.sizeof(h.struct_Before) == 4


def test_recovery_after_double_inclusion(tmp_path):
    (tmp_path / "twice.h").write_text("struct Ok { int x; };\ntypeof(int) bad;\nstruct Ok2 { int y; };\n")
    h = load_text(tmp_path, '#include "twice.h"\n#include "twice.h"\nstruct Last { int z; };\n')
    parse_errors = [d for d in diagnostics(h) if d.kind == "parse-error"]
    assert [(d.file, d.line) for d in parse_errors] == [(str(tmp_path / "twice.h"), 2)] * 2
    assert ctypes.sizeof(h.struct_Last) == 4 and ctypes.sizeof(h.struct_Ok2) == 4


# -- 22. typed() -------------------------------------------------------------------------------


HOST = {
    "x86_64": "x64",
    "amd64": "x64",
    "aarch64": "aarch64",
    "arm64": "aarch64",
    "riscv64": "riscv64",
    "loongarch64": "loongarch64",
}.get(platform.machine().lower())


def test_typed_on_every_architecture(types_h):
    h = types_h
    t = h.struct_Nested
    want = {"x": t.x.offset, "i": t.i.offset, "s2": t.s2.offset, "l": t.inner.offset + h.struct_Inner.l.offset}
    views = [X.typed(X.rdi, t), A.typed(A.x0, t), R.typed(R.a0, t), L.typed(L.a0, t)]
    for p in views:
        got = {"x": p.x.disp, "i": p.i.disp, "s2": p.s2.disp, "l": p.inner.l.disp}
        assert got == want
        assert (p.x.size, p.s2.size, p.inner.l.size) == (4, 2, 8)


@pytest.mark.skipif(HOST is None or sys.platform == "win32", reason="needs a POSIX host jita generates code for")
def test_execute_field_read(types_h):
    h = types_h
    a = Assembler(HOST)
    if HOST == "x64":
        a.mov(X.rax, X.typed(X.rdi, h.struct_Nested).inner.l)
    elif HOST == "aarch64":
        a.ldr(A.x0, A.typed(A.x0, h.struct_Nested).inner.l)
    elif HOST == "riscv64":
        a.ld(R.a0, R.typed(R.a0, h.struct_Nested).inner.l)
    else:
        a.ld.d(L.a0, L.typed(L.a0, h.struct_Nested).inner.l)
    a.ret()
    fn = a.function(ctypes.c_int64, ctypes.POINTER(h.struct_Nested))
    n = h.struct_Nested(x=1)
    n.inner.l = 123456789
    assert fn(ctypes.byref(n)) == 123456789


# -- review findings ---------------------------------------------------------------------


def test_preprocessing_numbers_are_single_tokens(tmp_path):
    # pcpp alone splits 8LL into 8L and L, and 2.5f into 2 . 5 f.
    h = load_text(
        tmp_path,
        """
        #define L -1
        #define f +1
        #define p +2
        struct S { char a[8LL]; };
        enum { BIGK = 0x100000000LL };
        #define RATE 2.5f
        #define BIGM 0x7fffffffffffffffLL
        #define HEXF 0x1p4
        #define DOT .5f
        #if 1LL && 0xffffffffffffffffULL > 0
        #define IF 1
        #endif
        """,
    )
    assert ctypes.sizeof(h.struct_S) == 8
    assert (h.BIGK, h.RATE, h.BIGM, h.HEXF, h.DOT, h.IF) == (1 << 32, 2.5, (1 << 63) - 1, 16.0, 0.5, 1)


def test_pragma_operator(tmp_path):
    h = load_text(
        tmp_path,
        """
        _Pragma("pack(push, 1)")
        struct A { char c; int i; };
        _Pragma("pack(pop)")
        #define PACKED _Pragma("pack(1)")
        PACKED
        struct B { char c; int i; };
        #pragma pack()
        struct C { char c; _Pragma("GCC diagnostic ignored \\"-Wx\\"") int i; };
        struct D { char c; _Pragma("pack(2)") int i; };
        struct E { char c; int i; };
        """,
    )
    assert [ctypes.sizeof(getattr(h, f"struct_{n}")) for n in "ABCE"] == [5, 5, 8, 8]
    assert "#pragma pack inside" in skips(h)["struct_D"].message


def test_unnamed_typedef_member(tmp_path):
    # GCC ignores `Inner;` (declares nothing), MSVC embeds it.
    h = load_text(tmp_path, "typedef struct Inner { int a; char b; } Inner;\nstruct MS { char c; Inner; int z; };\n")
    assert "Microsoft extension" in skips(h)["struct_MS"].message


def test_conditional_expression_types(tmp_path):
    h = load_text(tmp_path, "#define Z (0 ? 1u : -1)\n#define F (1 ? 2 : 3.0)\n#define S (1 ? -1 : 2)\n")
    assert (h.Z, h.F, h.S) == (4294967295, 2.0, -1)
    assert type(h.F) is float


@pytest.mark.parametrize("unsigned", [False, True])
def test_plain_char_signedness(tmp_path, monkeypatch, unsigned):
    from jita.cheader import prelude

    monkeypatch.setattr(prelude, "CHAR_UNSIGNED", unsigned)
    h = load_text(tmp_path, "#define E '\\xff'\n#define C ((char)200)\n#define S ((signed char)200)\n")
    assert (h.E, h.C, h.S) == ((255, 200, -56) if unsigned else (-1, -56, -56))


def test_linux_uapi_types(tmp_path):
    h = load_text(
        tmp_path,
        "#include <linux/types.h>\n"
        "struct U { __u8 a; __s16 b; __be32 c; __le16 d; __aligned_u64 e; __s64 f; __u32 g; __s8 h; };\n",
    )
    t = dict(h.struct_U._fields_)
    assert (t["a"], t["b"], t["c"], t["d"]) == (ctypes.c_uint8, ctypes.c_int16, ctypes.c_uint32, ctypes.c_uint16)
    assert (t["e"], t["f"], t["g"], t["h"]) == (ctypes.c_uint64, ctypes.c_int64, ctypes.c_uint32, ctypes.c_int8)
    assert ctypes.sizeof(h.struct_U) == 40 and h.struct_U.e.offset == 16


def test_one_diagnostic_per_skipped_definition(tmp_path):
    h = load_text(tmp_path, "struct GA { char c; __attribute__((__aligned__(sizeof(int)))) char d; };\n")
    skipped = [d for d in diagnostics(h) if d.severity == "skip"]
    assert [d.name for d in skipped] == ["struct_GA"]
    assert "1 skipped" in (h.__doc__ or "")


def test_dependency_messages_do_not_nest(tmp_path):
    h = load_text(
        tmp_path,
        "struct A { __int128 w; };\nstruct B { struct A a; };\nstruct C { struct B b; };\ntypedef struct C C_t;\n"
        "struct D { C_t c; };\n",
    )
    s = skips(h)
    assert s["struct_B"].message == "field a: struct_A was skipped (field w: __int128 has no ctypes type)"
    assert s["struct_C"].message == "field b: struct_B was skipped (struct_A: field w: __int128 has no ctypes type)"
    assert s["C_t"].message == "struct_C was skipped (struct_A: field w: __int128 has no ctypes type)"
    assert s["struct_D"].message.count("was skipped") == 1


def test_attribute_on_a_forward_declaration(tmp_path):
    # GCC ignores it; the definition decides.
    h = load_text(tmp_path, "struct __attribute__((aligned(8))) FA;\nstruct FA { char c; };\n")
    assert not skips(h) and ctypes.sizeof(h.struct_FA) == 1
    assert any("ignored on a declaration without a body" in d.message for d in notes(h, "unsupported"))


@pytest.mark.skipif(sys.platform in ("win32", "darwin"), reason="plain char is signed in the Windows and Apple ABIs")
@pytest.mark.parametrize("unsigned", [False, True])
def test_char_unsigned_macro(tmp_path, monkeypatch, unsigned):
    from jita.cheader import prelude

    monkeypatch.setattr(prelude, "CHAR_UNSIGNED", unsigned)
    assert ("__CHAR_UNSIGNED__ 1" in prelude.predefined_macros()) == unsigned
    h = load_text(tmp_path, "#ifdef __CHAR_UNSIGNED__\n#define U 1\n#else\n#define U 0\n#endif\n")
    assert h.U == int(unsigned)


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="the Linux UAPI bit field order macros")
def test_linux_bitfield_order(tmp_path):
    # The pattern of linux/if_packet.h; <asm/byteorder.h> is not read.
    h = load_text(
        tmp_path,
        """
        #include <asm/byteorder.h>
        struct fanout_args {
        #if defined(__LITTLE_ENDIAN_BITFIELD)
            unsigned short id;
            unsigned short type_flags;
        #else
            unsigned short type_flags;
            unsigned short id;
        #endif
        };
        """,
    )
    first = "id" if sys.byteorder == "little" else "type_flags"
    assert getattr(h.struct_fanout_args, first).offset == 0


def test_conditional_branch_not_taken(tmp_path):
    h = load_text(
        tmp_path,
        """
        #define DIV(a,b) ((b) ? (a)/(b) : 0)
        enum { EA = DIV(10, 0), EB = DIV(10, 2) };
        struct S { char a[DIV(8, 0) + 4]; };
        #define XA (DIV(7, 0))
        #define XB (1 ? 3 : 1 << -1)
        #define XF (1 ? 2.0 : 1.0 / 0)
        #define XU (1 ? 5 : 1u / 0)
        """,
    )
    assert (h.EA, h.EB, ctypes.sizeof(h.struct_S), h.XA, h.XB, h.XF, h.XU) == (0, 5, 4, 0, 3, 2.0, 5)
    assert not skips(h)


def test_pragma_operator_survives_recovery(tmp_path):
    # _Pragma becomes a #pragma line, so recovery from the broken
    # declaration next to it cannot take it out.
    h = load_text(
        tmp_path,
        """
        #define BEGIN _Pragma("pack(push, 1)")
        #define END _Pragma("pack(pop)")
        BEGIN
        typeof(int) x;
        struct A { char c; int i; };
        END
        struct B { char c; int i; };
        struct C { char c; _Pragma("GCC diagnostic ignored \\"-Wx\\"") int i; };
        """,
    )
    assert (ctypes.sizeof(h.struct_A), h.struct_A.i.offset, ctypes.sizeof(h.struct_B)) == (5, 1, 8)
    assert ctypes.sizeof(h.struct_C) == 8
    assert [d.line for d in skips(h).values()] == [5]


def test_binary_literals(tmp_path):
    h = load_text(tmp_path, "#if 0b11 == 3 && 0B11u > 2\n#define IF 1\n#endif\n#define B 0b101\nenum { E = 0b11 };\n")
    assert (h.IF, h.B, h.E) == (1, 5, 3)


def test_msvc_pragma_operator():
    h = load(FIXTURES / "msvc.h")
    assert (ctypes.sizeof(h.struct_A), h.struct_A.i.offset) == (5, 1)
    assert (ctypes.sizeof(h.struct_B), h.struct_B.c.offset) == (8, 4)
    assert h.unused._argtypes_ == (ctypes.c_int,)
    assert (ctypes.sizeof(h.struct_C), h.struct_C.i.offset) == (6, 2)
    assert ctypes.sizeof(h.struct_D) == 8
    assert list(skips(h)) == ["struct_E"] and "#pragma pack inside" in skips(h)["struct_E"].message
