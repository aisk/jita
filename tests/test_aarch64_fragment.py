"""aarch64 fragments: register and label holes, patch replay, extern slots.

Every instance is compared with the same instruction written directly with
the concrete registers: the bytes must be identical, and llvm-mc (when
installed) must decode both to the same text.
"""

from dataclasses import fields, is_dataclass, replace

import pytest
from oracle import aarch64_disassemble, requires_aarch64_disassembler

import jita.aarch64 as A
from jita import Assembler, EncodeError, Extern, Fragment, Hole, Label, label
from jita.aarch64 import *  # noqa: F403
from jita.aarch64.patch import REL19, REL26, RegField
from jita.tools.listing import listing

D, S, T, U = Hole.gp64("d"), Hole.gp64("s"), Hole.gp64("t"), Hole.gp64("u")
E, F, G = Hole.gp32("e"), Hole.gp32("f"), Hole.gp32("g")
P, Q, R = Hole.fp64("p"), Hole.fp64("q"), Hole.fp64("r")
SP_, SQ, SR = Hole.fp32("sp_"), Hole.fp32("sq"), Hole.fp32("sr")
L = Hole.label("lbl")

_CTOR = {("gp", 8): gp64, ("gp", 4): gp32, ("fp", 8): fp64, ("fp", 4): fp32}  # noqa: F405


def subst(op, values):
    if isinstance(op, Hole):
        return values[op.name]
    if is_dataclass(op):
        changes = {f.name: values[x.name] for f in fields(op) if isinstance(x := getattr(op, f.name), Hole)}
        return replace(op, **changes)
    return op


# (mnemonic function, operands with holes). Together they put a hole in
# every register field: Rd/Rt (bit 0), Rn and the memory base (bit 5),
# Ra/Rt2 (bit 10), Rm and the memory index (bit 16).
# Hole arithmetic builds addresses for the arch of the current assembler.
with Assembler(A):
    # fmt: off
    CASES = [
        (add, (D, S, T)),  # noqa: F405
        (add, (E, F, G)),  # noqa: F405
        (add, (D, S, 16)),  # noqa: F405
        (add, (D, x1, T)),  # noqa: F405
        (sub, (D, S, T)),  # noqa: F405
        (sub, (E, F, 100)),  # noqa: F405
        (mov, (D, S)),  # noqa: F405
        (neg, (E, F)),  # noqa: F405
        (ldr, (D, mem[S + 8])),  # noqa: F405
        (ldr, (E, mem[S])),  # noqa: F405
        (ldr, (D, mem[S + T])),  # noqa: F405
        (ldr, (x3, mem[x4 + T])),  # noqa: F405
        (ldr, (P, mem[S + 16])),  # noqa: F405
        (ldr, (D, mem.post[S, 8])),  # noqa: F405
        (ldrb, (E, mem[S + 1])),  # noqa: F405
        (ldr, (D, mem[S - 8])),  # noqa: F405
        (str_, (D, mem[S + 8])),  # noqa: F405
        (str_, (E, mem.pre[S - 16])),  # noqa: F405
        (str_, (SP_, mem[S + 4])),  # noqa: F405
        (ldp, (D, T, mem[S + 16])),  # noqa: F405
        (stp, (D, T, mem.pre[S - 32])),  # noqa: F405
        (ldp, (P, Q, mem[S])),  # noqa: F405
        (stp, (E, F, mem[S + 8])),  # noqa: F405
        (mul, (D, S, T)),  # noqa: F405
        (madd, (D, S, T, U)),  # noqa: F405
        (msub, (E, F, G, w9)),  # noqa: F405
        (csel, (D, S, T, "ne")),  # noqa: F405
        (csel, (E, F, G, "lt")),  # noqa: F405
        (fadd, (P, Q, R)),  # noqa: F405
        (fadd, (SP_, SQ, SR)),  # noqa: F405
        (fmadd, (P, Q, R, d7)),  # noqa: F405
        (fcvtzs, (D, Q)),  # noqa: F405
        (scvtf, (P, E)),  # noqa: F405
    ]
    # fmt: on


def _holes(ops):
    out = {}
    for op in ops:
        if isinstance(op, Hole):
            out[op.name] = op
        elif is_dataclass(op):
            for f in fields(op):
                if isinstance(x := getattr(op, f.name), Hole):
                    out[x.name] = x
    return list(out.values())


def variants(holes):
    """31 value sets: hole j gets register (n + 7*j) % 31 in variant n, so
    every hole takes every number 0..30 and no two holes are equal."""
    for n in range(31):
        yield {h.name: _CTOR[h.regclass, h.size]((n + 7 * j) % 31) for j, h in enumerate(holes)}


def _case_id(case):
    fn, ops = case
    return f"{fn.__name__}({', '.join(map(str, ops))})"


def build(fn, ops):
    frag = Fragment(A)
    fn(*ops, asm=frag)
    inst, direct = Assembler(A), Assembler(A)
    for values in variants(_holes(ops)):
        frag.instantiate(inst, **values)
        fn(*(subst(op, values) for op in ops), asm=direct)
    return frag, inst, direct


@pytest.mark.parametrize("case", CASES, ids=[_case_id(c) for c in CASES])
def test_register_holes_in_every_field(case):
    frag, inst, direct = build(*case)
    assert len(frag) == 4
    assert all(isinstance(p.kind, RegField) for p in frag.cur.patches)
    assert inst.cur.buf == direct.cur.buf


@requires_aarch64_disassembler
@pytest.mark.parametrize("case", CASES, ids=[_case_id(c) for c in CASES])
def test_register_holes_against_llvm_mc(case):
    _, inst, direct = build(*case)
    got = aarch64_disassemble(bytes(inst.cur.buf))
    assert got == aarch64_disassemble(bytes(direct.cur.buf))
    for line, values in zip(got, variants(_holes(case[1]))):
        assert "<invalid>" not in line and "udf" not in line
        for reg in values.values():
            assert str(reg) in line.replace(",", " ").replace("[", " ").replace("]", " ").split()


def test_register_hole_listing():
    frag = Fragment(A)
    madd(D, S, T, U, asm=frag)  # noqa: F405
    names = [p.kind.name for p in frag.cur.patches]
    assert names == ["rd", "rn", "rn.hi", "rm", "ra"]
    a = Assembler(A)
    frag.instantiate(a, d=x1, s=x2, t=x3, u=x4)  # noqa: F405
    assert "madd x1, x2, x3, x4" in listing(a)


@pytest.mark.parametrize("reg", ["xzr", "sp"])
def test_register_31_rejected(reg):
    frag = Fragment(A)
    add(D, S, 1, asm=frag)  # noqa: F405
    a = Assembler(A)
    with pytest.raises(EncodeError, match="register 31" if reg == "xzr" else "needs a register"):
        frag.instantiate(a, d=getattr(A, reg), s=x1)  # noqa: F405
    with pytest.raises(EncodeError, match="register 31"):
        frag.instantiate(a, d=x1, s=xzr)  # noqa: F405
    assert a.pos() == 0
    frag = Fragment(A)
    with frag:
        ldr(x0, mem[S + 8])  # noqa: F405
    with pytest.raises(EncodeError, match="register 31"):
        frag.instantiate(a, s=xzr)  # noqa: F405
    with pytest.raises(EncodeError, match="needs a register"):
        frag.instantiate(a, s=sp)  # noqa: F405
    # FP register 31 is an ordinary register.
    frag = Fragment(A)
    fadd(P, P, d1, asm=frag)  # noqa: F405
    frag.instantiate(a, p=d31)  # noqa: F405
    assert a.cur.buf.hex() == "ff2b611e"


def test_register_hole_type_errors():
    frag = Fragment(A)
    add(D, S, 1, asm=frag)  # noqa: F405
    with pytest.raises(EncodeError, match="needs a register"):
        frag.instantiate(Assembler(A), d=w1, s=x1)  # noqa: F405
    with pytest.raises(EncodeError, match="size mismatch"):
        add(D, E, 1, asm=Fragment(A))  # noqa: F405
    with pytest.raises(EncodeError, match="not an aarch64 register hole"):
        add(D, Hole.xmm("v"), 1, asm=Fragment(A))  # noqa: F405
    with pytest.raises(EncodeError, match="only be used inside a Fragment"):
        add(D, x1, 1, asm=Assembler(A))  # noqa: F405


def test_immediate_holes_are_rejected():
    with pytest.raises(EncodeError, match="immediate holes are not supported on aarch64"):
        add(x0, x1, Hole.imm32("k"), asm=Fragment(A))  # noqa: F405
    with pytest.raises(EncodeError, match="immediate holes are not supported on aarch64"):
        movz(x0, Hole.imm16("k"), asm=Fragment(A))  # noqa: F405


def test_hole_addresses():
    with Fragment(A):
        assert str(mem[S + 8]) == "[s, #8]"  # noqa: F405
        assert str(mem[8 + S]) == "[s, #8]"  # noqa: F405
        assert str(mem[S - 8]) == "[s, #-8]"  # noqa: F405
        assert str(mem[x0 + S]) == "[x0, s]"  # noqa: F405
        assert str(mem[S + T]) == "[s, t]"  # noqa: F405
        assert str(mem.pre[S + 16]) == "[s, #16]!"  # noqa: F405
        assert str(mem.post[S, 16]) == "[s], #16"  # noqa: F405
        with pytest.raises(EncodeError, match="index << n"):
            S * 8
        with pytest.raises(EncodeError, match="memory base"):
            E + 8
        with pytest.raises(EncodeError, match="cannot be a memory base or index"):
            mem[x0 + E]  # noqa: F405


def test_writeback_overlap_with_the_same_hole():
    with pytest.raises(EncodeError, match="unpredictable"):
        ldr(D, mem.post[D, 8], asm=Fragment(A))  # noqa: F405
    with pytest.raises(EncodeError, match="unpredictable"):
        ldp(D, D, mem[S], asm=Fragment(A))  # noqa: F405


# -- labels ----------------------------------------------------------------------


def _label_code(local, ext, a=None):
    """The code of the label fragment, with `local` the loop label."""
    add(x0, x0, 1, asm=a)  # noqa: F405
    cbnz(x0, local, asm=a)  # noqa: F405
    b("out", asm=a)  # noqa: F405
    bl(ext, asm=a)  # noqa: F405
    ldr(x1, "lit", asm=a)  # noqa: F405
    tbz(x1, 3, local, asm=a)  # noqa: F405
    adr(x2, local, asm=a)  # noqa: F405


EXT = 0x10000 + 0x2000
# Two instances at 0x10000 and 0x1001c, then ret at 0x10038 ("out"), a NOP
# of alignment padding and the literal at 0x10040 ("lit").
WANT = (
    "00040091" "e0ffffb5" "0c000014" "fd070094" "81010058" "61ff1f36" "42ffff10"
    "00040091" "e0ffffb5" "05000014" "f6070094" "a1000058" "61ff1f36" "42ffff10"
    "c0035fd6" "1f2003d5" "8877665544332211"
)
WANT_TEXT = [
    "add x0, x0, #1", "cbnz x0, #-4", "b #48", "bl #8180", "ldr x1, #48", "tbz w1, #3, #-20", "adr x2, #-24",
    "add x0, x0, #1", "cbnz x0, #-4", "b #20", "bl #8152", "ldr x1, #20", "tbz w1, #3, #-20", "adr x2, #-24",
    "ret",
]  # fmt: skip


def _label_target(build):
    a = Assembler(A)
    with a:
        build(a)
        label("out")
        ret()  # noqa: F405
        a.align(8)
        label("lit")
        a.qword(0x1122334455667788)
    return a.link(base=0x10000, externs={"f": EXT}).data


def _label_fragment():
    f = Fragment(A)
    with f:
        _label_code(label(), Extern("f"))
    return f


def test_labels_in_fragment():
    f = _label_fragment()
    assert all(isinstance(p.target, (Label, Extern)) for p in f.cur.patches)
    insts = []
    data = _label_target(lambda a: insts.extend([f.instantiate(), f.instantiate()]))
    assert data.hex() == WANT
    assert insts[1].start == 28

    def direct(a):
        for _ in range(2):
            _label_code(a.label(), Extern("f"), a)

    assert _label_target(direct) == data


@requires_aarch64_disassembler
def test_labels_in_fragment_against_llvm_mc():
    f = _label_fragment()
    data = _label_target(lambda a: (f.instantiate(), f.instantiate()))
    assert aarch64_disassemble(data[: 15 * 4]) == WANT_TEXT


def test_reproducer_from_review():
    # A local label and a named outside label used to come out as udf.
    f = Fragment(A)
    with f:
        lbl = label()
        add(x0, x0, 1)  # noqa: F405
        cbnz(x0, lbl)  # noqa: F405
        b("out")  # noqa: F405
    a = Assembler(A)
    with a:
        f.instantiate()
        label("out")
        ret()  # noqa: F405
    assert a.link().data.hex() == "00040091" "e0ffffb5" "01000014" "c0035fd6"


def test_label_hole_shared_by_two_instances():
    f = Fragment(A)
    with f:
        cbz(D, L)  # noqa: F405
        ldr(S, L)  # noqa: F405
        b(L)  # noqa: F405
    assert [p.kind for p in f.cur.patches if p.target is L] == [REL19, REL19, REL26]
    a = Assembler(A)
    with a:
        exit_ = Label()
        f.instantiate(d=x1, s=x2, lbl=exit_)  # noqa: F405
        f.instantiate(d=x3, s=x4, lbl="back")  # noqa: F405
        label(exit_)
        label("back")
        ret()  # noqa: F405
    assert a.link().data.hex() == (
        "c10000b4" "a2000058" "04000014"  # cbz x1, +24; ldr x2, +20; b +16
        "630000b4" "44000058" "01000014"  # cbz x3, +12; ldr x4, +8; b +4
        "c0035fd6"
    )


def test_nested_fragment_passes_holes_through():
    inner = Fragment(A)
    with inner:
        add(D, S, 1)  # noqa: F405
        cbz(D, L)  # noqa: F405
    outer = Fragment(A)
    d2, l2 = Hole.gp64("d2"), Hole.label("l2")
    with outer:
        inner.instantiate(d=d2, s=x5, lbl=l2)  # noqa: F405
        inner.instantiate(d=x6, s=d2, lbl=l2)  # noqa: F405
    a = Assembler(A)
    with a:
        outer.instantiate(d2=x9, l2="end")  # noqa: F405
        label("end")
    direct = Assembler(A)
    with direct:
        add(x9, x5, 1)  # noqa: F405
        cbz(x9, "end")  # noqa: F405
        add(x6, x9, 1)  # noqa: F405
        cbz(x6, "end")  # noqa: F405
        label("end")
    assert a.link().data == direct.link().data
