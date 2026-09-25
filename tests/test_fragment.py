"""Fragments with holes.

The encoding tests instantiate a one-instruction fragment with every
register number in every hole and check the result two ways: objdump must
decode each instance to the instruction written with the concrete operands,
and where GNU as can express the fixed-length form jita uses (a REX prefix
via `{rex}`, disp8/disp32 via `{disp8}`/`{disp32}`, 3 byte VEX via `{vex3}`)
the bytes must be identical. GNU as cannot force a SIB byte without an
index register, so `[hole]` with a base other than rsp/r12 is only checked
through objdump, which prints it as `[base+riz*1+disp]`.
"""

import ctypes
import platform
import re

import pytest
from oracle import assemble, disassemble, normalize, requires_oracle

import jita.x64 as x64
from jita import Assembler, EncodeError, Extern, Fragment, Hole, JitaError, Label, LinkError, label
from jita.core import ABS64, Instance, REL32
from jita.core.patch import BitsKind, ImmKind
from jita.tools.listing import listing
from jita.x64 import *  # noqa: F403

# `test` is an x64 mnemonic; keep pytest from collecting it.
insn_test = test  # noqa: F405
del test

needs_x64_host = pytest.mark.skipif(
    platform.machine().lower() not in ("x86_64", "amd64") or platform.system() == "Windows",
    reason="needs an x86-64 SysV host",
)

D, S, IDX = Hole.gp64("d"), Hole.gp64("s"), Hole.gp64("idx")
E, F = Hole.gp32("e"), Hole.gp32("f")
W, V = Hole.gp16("w"), Hole.gp16("v")
B, C = Hole.gp8("b"), Hole.gp8("c")
X1, X2, X3 = Hole.xmm("x1"), Hole.xmm("x2"), Hole.xmm("x3")
Y1, Y2, Y3 = Hole.ymm("y1"), Hole.ymm("y2"), Hole.ymm("y3")
I8, I16, I32, I64 = Hole.imm8("i8"), Hole.imm16("i16"), Hole.imm32("i32"), Hole.imm64("i64")
L = Hole.label("lbl")

_CTOR = {("gp", 1): gp8, ("gp", 2): gp16, ("gp", 4): gp32, ("gp", 8): gp64, ("xmm", 16): xmm, ("ymm", 32): ymm}  # noqa: F405
# Immediate values large enough that GNU as picks the same immediate size.
_IMM = {1: 0x5A, 2: 0x1234, 4: 0x12345678, 8: 0x123456789ABCDEF0}


def frag_of(build, **kw) -> Fragment:
    f = Fragment(x64, **kw)
    with f:
        build()
    return f


def variants(frag: Fragment):
    """16 value sets: hole j gets register (n + 5*j) % 16 in variant n."""
    regs = [h for h in frag.holes.values() if h.kind == "reg"]
    index = {p.target for p in frag.cur.patches if p.kind.name == "sib.index"}
    out = []
    for n in range(16 if regs else 1):
        vals = {}
        for j, h in enumerate(regs):
            code = (n + 5 * j) % 16
            if h in index and code == 4:
                code = 5  # rsp cannot be an index
            vals[h.name] = _CTOR[h.regclass, h.size](code)
        for h in frag.holes.values():
            if h.kind == "imm":
                vals[h.name] = _IMM[h.size]
        out.append(vals)
    return out


def insn_text(mnemonic, ops):
    return mnemonic + (" " + ", ".join(str(o) for o in ops) if ops else "")


def canon(line: str) -> str:
    """objdump text of a fixed-length form -> text of the plain instruction."""
    line = re.sub(r"^rex(\.\w+)? ", "", line)
    line = line.replace("+riz*1", "").replace("+0x0]", "]")
    return sym(line.replace("movabs ", "mov "))


def sym(line: str) -> str:
    """xchg is symmetric; objdump may print its operands the other way."""
    if line.startswith("xchg "):
        return "xchg " + ",".join(sorted(line[5:].split(",")))
    return line


# (id, build, check bytes against GNU as). "acc" skips the variants where
# GNU as picks a shorter accumulator form because the register is number 0.
CASES = [
    ("mov r64 r64", lambda: mov(D, S), True),  # noqa: F405
    ("mov r32 r32", lambda: mov(E, F), True),  # noqa: F405
    ("mov r16 r16", lambda: mov(W, V), True),  # noqa: F405
    ("mov r8 r8", lambda: mov(B, C), True),  # noqa: F405
    ("add r64 [base+8]", lambda: add(D, qword[S + 8]), True),  # noqa: F405
    ("mov [base] r64", lambda: mov(qword[S], D), True),  # noqa: F405
    ("mov r32 [base+disp32]", lambda: mov(E, dword[S + 0x1000]), True),  # noqa: F405
    ("mov r8 [base-128]", lambda: mov(B, byte[S - 128]), True),  # noqa: F405
    ("lea [base+idx*4+16]", lambda: lea(D, ptr[S + IDX * 4 + 16]), True),  # noqa: F405
    ("mov [base+idx*8] disp32", lambda: mov(rax, qword[S + IDX * 8 + 1000]), True),  # noqa: F405
    ("mov [rbp+idx*8]", lambda: mov(rax, qword[rbp + IDX * 8]), True),  # noqa: F405
    ("mov [r13+idx*2]", lambda: mov(ecx, dword[r13 + IDX * 2]), True),  # noqa: F405
    ("mov [idx*8+disp]", lambda: mov(rax, qword[IDX * 8 + 0x100]), True),  # noqa: F405
    ("mov [base+rcx*8]", lambda: mov(rdx, qword[S + rcx * 8]), True),  # noqa: F405
    ("push r64", lambda: push(D), True),  # noqa: F405
    ("pop r64", lambda: pop(D), True),  # noqa: F405
    ("bswap r32", lambda: bswap(E), True),  # noqa: F405
    ("inc r16", lambda: inc(W), True),  # noqa: F405
    ("movzx r32 r8", lambda: movzx(eax, B), True),  # noqa: F405
    ("movzx r64 [base]", lambda: movzx(D, byte[S]), True),  # noqa: F405
    ("xchg eax r32", lambda: xchg(eax, E), False),  # noqa: F405  # as picks 90+r
    ("mov r32 imm32", lambda: mov(E, I32), True),  # noqa: F405
    ("mov r64 imm32", lambda: mov(D, I32), True),  # noqa: F405
    ("mov r64 imm64", lambda: mov(D, I64), True),  # noqa: F405
    ("mov r8 imm8", lambda: mov(B, I8), True),  # noqa: F405
    ("mov r16 imm16", lambda: mov(W, I16), True),  # noqa: F405
    ("add r64 imm8", lambda: add(D, I8), True),  # noqa: F405
    ("add r32 imm32", lambda: add(E, I32), "acc"),  # noqa: F405
    ("add [base+idx] imm8", lambda: add(qword[S + IDX], I8), True),  # noqa: F405
    ("mov byte [base] imm8", lambda: mov(byte[S + 1], I8), True),  # noqa: F405
    ("shl r64 imm8", lambda: shl(D, I8), True),  # noqa: F405
    ("shl r32 cl", lambda: shl(E, cl), True),  # noqa: F405
    ("imul r r imm32", lambda: imul(D, S, I32), True),  # noqa: F405
    ("insn_test r64 imm32", lambda: insn_test(D, I32), "acc"),
    ("cmp r32 [base+idx*4]", lambda: cmp(E, dword[S + IDX * 4]), True),  # noqa: F405
    ("addsd xmm xmm", lambda: addsd(X1, X2), True),  # noqa: F405
    ("movsd xmm [base+8]", lambda: movsd(X1, qword[S + 8]), True),  # noqa: F405
    ("movd xmm r32", lambda: movd(X1, E), True),  # noqa: F405
    ("pshufd xmm xmm imm8", lambda: pshufd(X1, X2, I8), True),  # noqa: F405
    ("vaddps ymm ymm ymm", lambda: vaddps(Y1, Y2, Y3), True),  # noqa: F405
    ("vaddpd xmm reg only", lambda: vaddpd(X1, X2, xmm3), True),  # noqa: F405
    ("vaddsd [base+idx*8]", lambda: vaddsd(X1, X2, qword[S + IDX * 8]), True),  # noqa: F405
    ("vmovups [base] ymm", lambda: vmovups(yword[S + 32], Y1), True),  # noqa: F405
    ("vblendvps is4", lambda: vblendvps(X1, X2, oword[S], X3), True),  # noqa: F405
]


@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_fixed_length_and_fields(case):
    _, build, _ = case
    frag = frag_of(build)
    a = Assembler(x64)
    for vals in variants(frag):
        inst = frag.instantiate(a, **vals)
        assert inst.end - inst.start == len(frag)
    assert len(a.cur.buf) == len(frag) * len(a.cur.insns)
    assert a.cur.patches == []


@requires_oracle
@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_against_objdump_and_gas(case):
    _, build, gas = case
    frag = frag_of(build)
    a = Assembler(x64)
    for vals in variants(frag):
        frag.instantiate(a, **vals)
    code = bytes(a.cur.buf)
    texts = [insn_text(mn, ops) for _, _, mn, ops in a.cur.insns]
    assert [canon(line) for line in disassemble(code)] == [sym(normalize(t)) for t in texts]
    if not gas:
        return
    src, want = [], b""
    for (lo, hi, mn, ops), (_, _, _, fops) in zip(a.cur.insns, [frag.cur.insns[0]] * len(texts)):
        prefix = "{vex3} " if mn.startswith("v") and code[lo] == 0xC4 else "" if mn.startswith("v") else "{rex} "
        expressible = not (gas == "acc" and any(getattr(op, "code", None) == 0 for op in ops))
        for fop, op in zip(fops, ops):
            if isinstance(fop, MemExpr) and isinstance(fop.base, Hole):  # noqa: F405
                if op.index is None and op.base.code & 7 != 4:
                    expressible = False  # GNU as cannot force a SIB byte
                prefix += "{disp8} " if -128 <= op.disp <= 127 else "{disp32} "
        if expressible:
            src.append(prefix + insn_text(mn, ops))
            want += code[lo:hi]
    assert src, "no case expressible in GNU as"
    assert assemble("\n".join(src)) == want


def test_fixed_bytes():
    # Expected hex, checked against GNU as ({rex}, {disp8}, {vex3}) where it
    # can force the same form; the SIB forms without an index by hand.
    cases = [
        (lambda: mov(D, qword[S + 8]), dict(d=rbx, s=rdi), "488b5c2708"),  # noqa: F405
        (lambda: mov(D, qword[S + 8]), dict(d=r12, s=rsp), "4c8b642408"),  # noqa: F405
        (lambda: mov(D, qword[S]), dict(d=rax, s=r13), "498b442500"),  # noqa: F405
        (lambda: mov(E, F), dict(e=eax, f=ecx), "4089c8"),  # noqa: F405
        (lambda: push(D), dict(d=r15), "4157"),  # noqa: F405
        (lambda: mov(B, C), dict(b=sil, c=r9b), "4488ce"),  # noqa: F405
        (lambda: lea(D, ptr[S + IDX * 4 + 16]), dict(d=rax, s=rbp, idx=r12), "4a8d44a510"),  # noqa: F405
        (lambda: add(D, I32), dict(d=rcx, i32=1), "4881c101000000"),  # noqa: F405
        (lambda: add(D, I8), dict(d=rcx, i8=-1), "4883c1ff"),  # noqa: F405
        (lambda: mov(D, I64), dict(d=r9, i64=-1), "49b9ffffffffffffffff"),  # noqa: F405
        (lambda: addsd(X1, X2), dict(x1=xmm1, x2=xmm9), "f2410f58c9"),  # noqa: F405
        (lambda: vaddps(Y1, Y2, Y3), dict(y1=ymm0, y2=ymm1, y3=ymm2), "c4e17458c2"),  # noqa: F405
        (lambda: vaddps(Y1, Y2, ymm3), dict(y1=ymm8, y2=ymm15), "c50458c3"),  # noqa: F405
    ]
    for build, vals, want in cases:
        frag = frag_of(build)
        a = Assembler(x64)
        frag.instantiate(a, **vals)
        assert a.cur.buf.hex() == want, (frag.cur.insns, vals)


def test_patch_kinds_in_fragment():
    frag = frag_of(lambda: add(qword[S + IDX * 8], I8))  # noqa: F405
    kinds = {p.kind.name: p for p in frag.cur.patches}
    assert set(kinds) == {"rex.x", "rex.b", "sib.index", "sib.base", "imm8s"}
    assert isinstance(kinds["sib.index"].kind, BitsKind)
    assert isinstance(kinds["imm8s"].kind, ImmKind)
    assert kinds["sib.index"].target is IDX and kinds["imm8s"].target is I8
    assert frag.holes == {"s": S, "idx": IDX, "i8": I8}


# -- labels ------------------------------------------------------------------


def _loop_fragment():
    frag = Fragment(x64)
    with frag:
        label("top")
        dec(D)  # noqa: F405
        jnz.short("top")  # noqa: F405
        jz(L)  # noqa: F405
        lea(rax, ptr[rip + L + 4])  # noqa: F405
        call("outside")  # noqa: F405
    return frag


def test_local_labels_and_label_holes():
    frag = _loop_fragment()
    assert set(frag.holes) == {"d", "lbl"}
    a = Assembler(x64)
    with a:
        end = Label()
        i1 = frag.instantiate(d=rcx, lbl=end)
        i2 = frag.instantiate(d=r10, lbl="named_end")
        label(end)
        label("named_end")
        label("outside")
        ret()  # noqa: F405
    assert isinstance(i1, Instance) and i1.start == 0 and i2.start == len(frag)
    assert i1.labels["top"] is not i2.labels["top"]
    assert i1.labels["top"].offset == 0 and i2.labels["top"].offset == len(frag)
    assert i1.labels[L] is end and i2.labels[L] is a.named("named_end")
    assert i2.values == {"d": r10, "lbl": a.named("named_end")}  # noqa: F405
    # Fragment-local names are not symbols of the target assembler.
    img = a.link(base=0x1000)
    assert "top" not in img.symbols and img.symbols["outside"] == 2 * len(frag)


@requires_oracle
def test_labels_against_gas():
    frag = _loop_fragment()
    a = Assembler(x64)
    with a:
        frag.instantiate(d=rcx, lbl="end")
        frag.instantiate(d=r10, lbl="end")
        label("end")
        label("outside")
        ret()  # noqa: F405
    code = a.link(base=0).data
    text = """
    t1:
    {rex} dec rcx
    jnz t1
    {disp32} jz end
    lea rax, [rip+end+4]
    {disp32} call outside
    t2:
    dec r10
    jnz t2
    {disp32} jz end
    lea rax, [rip+end+4]
    {disp32} call outside
    end:
    outside:
    ret
    """
    assert assemble(text) == code


def test_label_hole_data_and_movabs():
    frag = Fragment(x64)
    with frag:
        mov(D, L)  # noqa: F405
        frag.qword(L)
        frag.dword(I32)
        frag.byte(I8)
        frag.qword(I64)
    a = Assembler(x64)
    target = a.label("t")
    frag.instantiate(a, d=rsi, lbl=target, i32=-2, i8=200, i64=1 << 63)  # noqa: F405
    img = a.link(base=0x10000)
    data = img.data
    assert data[:2] == bytes.fromhex("48be")
    assert int.from_bytes(data[2:10], "little") == 0x10000
    assert int.from_bytes(data[10:18], "little") == 0x10000
    assert data[18:22] == (-2 & 0xFFFFFFFF).to_bytes(4, "little")
    assert data[22] == 200
    assert int.from_bytes(data[23:31], "little") == 1 << 63
    assert {p.kind for p in a.cur.patches} == {ABS64}


def test_unbound_labels_and_externs_pass_through():
    ext = Extern("ext")
    frag = Fragment(x64)
    with frag:
        call(ext)  # noqa: F405
        jmp(frag.pc[3])  # noqa: F405
    a = Assembler(x64)
    frag.instantiate(a)
    targets = [p.target for p in a.cur.patches]
    assert targets == [ext, a.pc[3]]
    assert [p.kind for p in a.cur.patches] == [REL32, REL32]


def test_anonymous_local_label_and_short_jump():
    frag = Fragment(x64)
    top = Label()
    with frag:
        label(top)
        dec(E)  # noqa: F405
        jnz.short(top)  # noqa: F405
    a = Assembler(x64)
    i1 = frag.instantiate(a, e=eax)  # noqa: F405
    i2 = frag.instantiate(a, e=r11d)  # noqa: F405
    assert i1.labels[top].offset == 0 and i2.labels[top].offset == len(frag)
    assert a.link().data.hex() == "40ffc875fb" + "41ffcb75fb"


def test_nested_fragment():
    inner = frag_of(lambda: add(D, I8))  # noqa: F405
    outer = Fragment(x64)
    with outer:
        inner.instantiate(d=rax, i8=1)  # noqa: F405
        inner.instantiate(outer, d=r8, i8=2)  # noqa: F405
        sub(S, I8)  # noqa: F405
    a = Assembler(x64)
    outer.instantiate(a, s=rcx, i8=3)  # noqa: F405
    assert a.cur.buf.hex() == "4883c001" "4983c002" "4883e903"


def _inner_for_nesting() -> Fragment:
    d, s, i, k, lbl, n = (Hole.gp64("d"), Hole.gp64("s"), Hole.gp64("i"), Hole.imm32("k"),
                          Hole.label("l"), Hole.imm8("n"))  # fmt: skip
    return frag_of(
        lambda: (
            add(d, qword[s + i * 8 + 8]),  # noqa: F405
            mov(d, k),  # noqa: F405
            jz(lbl),  # noqa: F405
            lea(d, ptr[rip + lbl + 4]),  # noqa: F405
            shl(d, n),  # noqa: F405
            vaddsd(xmm1, xmm2, qword[s + 16]),  # noqa: F405
        )
    )


DD, II, KK, LL = Hole.gp64("dd"), Hole.gp64("ii"), Hole.imm32("kk"), Hole.label("ll")


def _outer_for_nesting(inner: Fragment) -> Fragment:
    outer = Fragment(x64)
    with outer:
        nop()  # noqa: F405
        inner.instantiate(d=DD, s=rbx, i=II, k=KK, l=LL, n=3)  # noqa: F405
        ret()  # noqa: F405
    return outer


def _direct_for_nesting(d, i, k, lbl) -> None:
    nop()  # noqa: F405
    add(d, qword[rbx + i * 8 + 8])  # noqa: F405
    mov(d, k)  # noqa: F405
    jz(lbl)  # noqa: F405
    lea(d, ptr[rip + lbl + 4])  # noqa: F405
    shl(d, 3)  # noqa: F405
    vaddsd(xmm1, xmm2, qword[rbx + 16])  # noqa: F405
    ret()  # noqa: F405


def test_nested_fragment_passes_holes_through():
    inner = _inner_for_nesting()
    outer = _outer_for_nesting(inner)
    assert outer.holes == {"dd": DD, "ii": II, "kk": KK, "ll": LL}
    # The inner hole fields are fields of the outer holes, same kinds.
    passed = {inner.holes[n]: h for n, h in (("d", DD), ("i", II), ("k", KK), ("l", LL))}
    inner_kinds = [
        (p.offset + 1, p.kind, passed[p.target], p.addend) for p in inner.cur.patches if p.target in passed
    ]
    outer_kinds = [(p.offset, p.kind, p.target, p.addend) for p in outer.cur.patches]
    key = lambda t: (t[0], t[1].name)  # noqa: E731
    assert sorted(outer_kinds, key=key) == sorted(inner_kinds, key=key)
    # The same bytes as instantiating the inner fragment directly.
    for d, i, k in [(rax, rcx, 1), (r9, r12, -5), (rsp, r15, 0x7FFFFFFF), (r13, rbp, 7)]:  # noqa: F405
        got, want = Assembler(x64), Assembler(x64)
        with got:
            outer.instantiate(dd=d, ii=i, kk=k, ll="end")
            label("end")
        with want:
            nop()  # noqa: F405
            inner.instantiate(d=d, s=rbx, i=i, k=k, l="end", n=3)  # noqa: F405
            ret()  # noqa: F405
            label("end")
        assert got.link().data == want.link().data
        assert listing(got) == listing(want)
    with pytest.raises(EncodeError, match="rsp"):
        outer.instantiate(Assembler(x64), dd=rax, ii=rsp, kk=0, ll=Label())  # noqa: F405


@requires_oracle
def test_nested_fragment_against_objdump():
    outer = _outer_for_nesting(_inner_for_nesting())
    for d, i, k in [(rax, rcx, 1), (r9, r12, -5), (r12, r13, 0x1234), (r15, rbp, 7)]:  # noqa: F405
        got, want = Assembler(x64), Assembler(x64)
        with got:
            outer.instantiate(dd=d, ii=i, kk=k, ll="end")
            label("end")
        with want:
            _direct_for_nesting(d, i, k, "end")
            label("end")
        # Branch and rip-relative targets print as addresses, which differ
        # with the instruction lengths; compare everything else.
        def text(asm):
            lines = [canon(line) for line in disassemble(asm.link().data)]
            return [ln for ln in lines if not ln.startswith("j") and "rip" not in ln]

        assert text(got) == text(want)


def test_nested_fragment_hole_type_mismatch():
    inner = _inner_for_nesting()
    ok = dict(d=DD, s=rbx, i=II, k=KK, l=LL, n=3)  # noqa: F405
    for name, bad in [
        ("d", Hole.gp32("e32")),
        ("d", Hole.xmm("x")),
        ("k", Hole.imm8("k8")),
        ("k", Hole.imm64("k64")),
        ("l", Hole.gp64("g")),
        ("d", Hole.label("l2")),
        ("n", Hole.imm32("n32")),
    ]:
        outer = Fragment(x64)
        with pytest.raises(TypeError, match="hole types differ"):
            inner.instantiate(outer, **{**ok, name: bad})
        assert outer.cur.buf == b"" and outer.holes == {}
    # Holes only pass through into another Fragment.
    with pytest.raises(EncodeError, match="another Fragment"):
        inner.instantiate(Assembler(x64), **ok)
    # The outer fragment cannot end up with two holes of one name.
    outer = Fragment(x64)
    outer.mov(Hole.gp64("dd"), rax)  # noqa: F405
    size = len(outer)
    with pytest.raises(EncodeError, match="two different holes"):
        inner.instantiate(outer, **ok)
    assert len(outer) == size and set(outer.holes) == {"dd"}


def test_label_hole_with_extern_value():
    ext = Extern("strlen")
    frag = Fragment(x64)
    with frag:
        jmp(L)  # noqa: F405
        call(L)  # noqa: F405
        mov(rax, L)  # noqa: F405
        call(qword[rip + L])  # noqa: F405
        mov(rcx, qword[rip + L])  # noqa: F405
        cmp(dword[rip + L], 5)  # noqa: F405
        frag.qword(L)
    a = Assembler(x64)
    inst = frag.instantiate(a, lbl=ext)
    assert inst.values == {"lbl": ext} and L not in inst.labels
    slot = a.extern_slots["strlen"]
    code = a.sections["code"].patches
    assert [(p.kind, p.target) for p in code] == [
        (REL32, ext), (REL32, ext), (ABS64, ext), (REL32, slot), (REL32, slot), (REL32, slot), (ABS64, ext),
    ]  # fmt: skip
    b = Assembler(x64)
    with b:
        jmp(ext)  # noqa: F405
        call(ext)  # noqa: F405
        mov(rax, ext)  # noqa: F405
        call(qword[rip + ext])  # noqa: F405
        mov(rcx, qword[rip + ext])  # noqa: F405
        cmp(dword[rip + ext], 5)  # noqa: F405
        b.qword(ext)
    externs = {"strlen": 0x2000}
    assert a.link(0x1000, externs).data == b.link(0x1000, externs).data
    assert "cmp dword ptr [rip+strlen@slot], 5" in listing(a)


def test_label_hole_with_extern_value_rejects_displacement():
    frag = frag_of(lambda: (nop(), mov(rax, qword[rip + L + 8])))  # noqa: F405
    a = Assembler(x64)
    with pytest.raises(EncodeError, match="pointer slot"):
        frag.instantiate(a, lbl=Extern("strlen"))
    assert a.cur.buf == b"" and a.extern_slots == {}
    frag.instantiate(a, lbl=a.label())  # a Label with a displacement is fine


def test_extern_value_through_nested_fragments():
    ext = Extern("strlen")
    inner = frag_of(lambda: (call(qword[rip + L]), jmp(L)))  # noqa: F405
    direct = Assembler(x64)
    with direct:
        call(qword[rip + ext])  # noqa: F405
        jmp(ext)  # noqa: F405
    want = direct.link(0x1000, {"strlen": 0x9000}).data
    # The Extern given to the inner fragment, or to an outer hole.
    via_value, via_hole = Fragment(x64), Fragment(x64)
    inner.instantiate(via_value, lbl=ext)
    inner.instantiate(via_hole, lbl=LL)
    assert via_value.extern_slots == {} and via_hole.holes == {"ll": LL}
    for outer, vals in ((via_value, {}), (via_hole, {"ll": ext})):
        a = Assembler(x64)
        outer.instantiate(a, **vals)
        assert list(a.extern_slots) == ["strlen"]
        assert a.link(0x1000, {"strlen": 0x9000}).data == want


@needs_x64_host
def test_exec_call_through_label_hole_extern():
    s = Hole.gp64("s")
    frag = frag_of(lambda: (mov(rdi, s), call(qword[rip + L])))  # noqa: F405
    a = Assembler(x64)
    with a:
        sub(rsp, 8)  # noqa: F405
        frag.instantiate(s=rsi, lbl=Extern("strlen"))  # noqa: F405
        add(rsp, 8)  # noqa: F405
        ret()  # noqa: F405
    addr = ctypes.cast(ctypes.CDLL(None).strlen, ctypes.c_void_p).value
    with a.load(externs={"strlen": addr}) as mod:
        fn = mod.function(ctypes.c_int64, ctypes.c_void_p, ctypes.c_char_p)
        assert fn(None, b"holes") == 5


# -- alignment ---------------------------------------------------------------


def test_alignment():
    frag = Fragment(x64, align=16)
    with frag:
        ret()  # noqa: F405
    a = Assembler(x64)
    frag.instantiate(a)
    a.bytes(b"\x90")
    with pytest.raises(LinkError, match="aligned to 16"):
        frag.instantiate(a)
    a.align(16)
    assert frag.instantiate(a).start == 16

    inner = Fragment(x64)
    with inner:
        nop()  # noqa: F405
        inner.align(8)
        ret()  # noqa: F405
    assert inner.alignment == 8 and len(inner) == 9
    b = Assembler(x64)
    b.bytes(b"\xcc" * 4)
    with pytest.raises(LinkError):
        inner.instantiate(b)


def test_nested_fragment_inherits_alignment():
    inner = Fragment(x64, align=16)
    with inner:
        ret()  # noqa: F405
    outer = Fragment(x64)
    with outer:
        inner.instantiate()
    assert outer.alignment == 16
    a = Assembler(x64)
    a.bytes(b"\x90" * 8)
    with pytest.raises(LinkError, match="aligned to 16"):
        outer.instantiate(a)

    # align() padding inside the inner fragment stays where it belongs: the
    # outer fragment needs the same alignment, so the padded instruction is
    # aligned in the final code too.
    padded = Fragment(x64)
    with padded:
        nop()  # noqa: F405
        padded.align(8)
        label("aligned")
        ret()  # noqa: F405
    outer = Fragment(x64)
    with outer:
        first = padded.instantiate()
        outer.align(8)
        second = padded.instantiate()
    assert [i.labels["aligned"].offset for i in (first, second)] == [8, 24]
    assert outer.alignment == 8
    b = Assembler(x64)
    b.bytes(b"\xcc" * 4)
    with pytest.raises(LinkError):
        outer.instantiate(b)
    b.align(8)
    outer.instantiate(b)
    rets = [i for i, x in enumerate(b.cur.buf) if x == 0xC3]
    assert rets == [16, 32]


def test_instance_end_is_fixed_at_instantiation():
    frag = frag_of(lambda: push(D))  # noqa: F405
    a = Assembler(x64)
    inst = frag.instantiate(a, d=rbx)  # noqa: F405
    with frag:
        pop(D)  # noqa: F405
    assert (inst.start, inst.end) == (0, 2)
    assert frag.instantiate(a, d=rbx).end == 6  # noqa: F405


def test_extern_slot_method_is_rejected_in_a_fragment():
    frag = Fragment(x64)
    with pytest.raises(JitaError, match=r"qword\[rip \+ ext\]"):
        frag.extern_slot(Extern("strlen"))
    assert list(frag.sections) == ["code"] and frag.extern_slots == {}


# -- listing -----------------------------------------------------------------


def test_listing_of_instances():
    frag = _loop_fragment()
    a = Assembler(x64)
    with a:
        frag.instantiate(d=rcx, lbl="end")
        frag.instantiate(d=r10, lbl="end")
        label("end")
        label("outside")
        ret()  # noqa: F405
    text = listing(a)
    assert "dec rcx" in text and "dec r10" in text
    assert "jz end" in text and "lea rax, [rip+end+4]" in text
    assert "jnz.short .L1" in text and "jnz.short .L2" in text
    assert "Hole" not in text and " d," not in text
    ftext = listing(frag)
    assert "dec d" in ftext and "modrm.rm -> d" in ftext and "jz lbl" in ftext


# -- errors ------------------------------------------------------------------


def test_hole_constructors():
    with pytest.raises(TypeError, match="typed constructor"):
        Hole("x")
    with pytest.raises(TypeError, match="identifier"):
        Hole.gp64("not a name")
    assert repr(Hole.gp8("q")) == "Hole.gp8('q')" and repr(Hole.ymm("q")) == "Hole.ymm('q')"
    assert Hole.imm8("q").range == (-128, 255)
    assert Hole.imm64("q").range == (-(1 << 63), (1 << 64) - 1)
    h = Hole.gp64("q")
    assert h != Hole.gp64("q") and h == h and len({h, Hole.gp64("q")}) == 2


def test_instantiate_errors():
    frag = frag_of(lambda: (add(D, qword[S + IDX * 8]), add(B, I8), jmp(L)))  # noqa: F405
    ok = dict(d=rax, s=rbx, idx=rcx, b=dl, i8=1, lbl=Label())  # noqa: F405
    a = Assembler(x64)
    with pytest.raises(TypeError, match="missing value for hole 'lbl'"):
        frag.instantiate(a, **{k: v for k, v in ok.items() if k != "lbl"})
    with pytest.raises(TypeError, match="no hole named 'extra'"):
        frag.instantiate(a, **ok, extra=1)
    for name, bad in [
        ("d", eax),  # noqa: F405
        ("d", xmm0),  # noqa: F405
        ("d", 5),
        ("b", ah),  # noqa: F405
        ("b", ax),  # noqa: F405
        ("i8", 256),
        ("i8", -129),
        ("i8", "x"),
        ("i8", True),
        ("lbl", 5),
        ("lbl", rax),  # noqa: F405
    ]:
        with pytest.raises(EncodeError):
            frag.instantiate(a, **{**ok, name: bad})
    # The add r8, imm8 form takes -128..255 but not beyond.
    with pytest.raises(EncodeError, match="rsp"):
        frag.instantiate(a, **{**ok, "idx": rsp})  # noqa: F405
    with pytest.raises(LinkError, match="another assembler"):
        frag.instantiate(a, **{**ok, "lbl": Assembler(x64).named("x")})
    assert a.cur.buf == b"" and a.cur.patches == [] and a.cur.insns == []
    frag.instantiate(a, **{**ok, "idx": r12})  # noqa: F405
    assert len(a.cur.buf) == len(frag)


def test_imm_use_narrows_range():
    frag = frag_of(lambda: add(D, I8))  # noqa: F405  # 83 /0 ib, sign extended
    a = Assembler(x64)
    frag.instantiate(a, d=rax, i8=-128)  # noqa: F405
    with pytest.raises(EncodeError, match="out of range -128..127"):
        frag.instantiate(a, d=rax, i8=200)  # noqa: F405
    frag = frag_of(lambda: add(D, I32))  # noqa: F405  # sign extended imm32
    with pytest.raises(EncodeError, match="out of range"):
        frag.instantiate(a, d=rax, i32=0xFFFFFFFF)  # noqa: F405
    frag = frag_of(lambda: add(E, I32))  # noqa: F405
    frag.instantiate(a, e=eax, i32=0xFFFFFFFF)  # noqa: F405


def test_encode_errors():
    def fails(build, match):
        f = Fragment(x64)
        with f, pytest.raises(EncodeError, match=match):
            build()
        assert f.cur.buf == b"" and f.holes == {}

    fails(lambda: mov(eax, I8), "no form with a 8 bit immediate")  # noqa: F405
    fails(lambda: add(rax, I64), "no form with a 64 bit immediate")  # noqa: F405
    fails(lambda: mov(ah, B), "bad mix")  # noqa: F405
    fails(lambda: mov(ah, byte[S]), "REX")  # noqa: F405
    fails(lambda: shl(D, B), "bad operand mode")  # noqa: F405  # count must be cl
    with pytest.raises(EncodeError, match="rip"):
        qword[rip + S]  # noqa: F405
    with pytest.raises(EncodeError, match="only gp64 holes"):
        qword[E]  # noqa: F405
    with pytest.raises(EncodeError, match="only gp64 holes"):
        with Assembler(x64):
            qword[rax + E * 2]  # noqa: F405
    with pytest.raises(EncodeError, match="mixed address sizes"):
        dword[eax + S]  # noqa: F405
    with pytest.raises(EncodeError, match="cannot be part of an address"):
        qword[rax + I32]  # noqa: F405
    # Holes need a Fragment.
    a = Assembler(x64)
    with pytest.raises(EncodeError, match="inside a Fragment"):
        mov(D, rax, asm=a)  # noqa: F405
    with pytest.raises(EncodeError, match="inside a Fragment"):
        a.qword(L)
    assert a.cur.buf == b""
    # Hole names are unique within a fragment.
    f = Fragment(x64)
    with f:
        mov(D, rax)  # noqa: F405
        with pytest.raises(EncodeError, match="two different holes"):
            mov(Hole.gp64("d"), rax)  # noqa: F405
        with pytest.raises(EncodeError, match="data value"):
            f.dword(I8)


def test_fragment_restrictions():
    f = Fragment(x64)
    with pytest.raises(JitaError, match="one section"):
        f.section("data")
    with pytest.raises(JitaError):
        f.link()
    with pytest.raises(JitaError):
        f.load()
    with pytest.raises(JitaError, match="itself"):
        f.instantiate(f)
    assert bool(f) and len(f) == 0
    f.add(D, 1)  # empty fragments are truthy, so methods emit into them
    assert len(f) == 4


def test_declare_unused_hole():
    k = Hole.imm32("k")
    f = Fragment(x64)
    f.declare(k)
    f.mov(D, rax)  # noqa: F405
    a = Assembler(x64)
    f.instantiate(a, d=rbx, k=7)  # noqa: F405
    assert a.cur.buf.hex() == "4889c3"


def test_hole_arithmetic():
    m = S + IDX * 8 + 16
    assert (m.base, m.index, m.scale, m.disp) == (S, IDX, 8, 16)
    m = 8 * IDX + S
    assert (m.base, m.index, m.scale) == (S, IDX, 8)
    m = rax + S  # noqa: F405
    assert (m.base, m.index) == (rax, S)  # noqa: F405
    m = S + rsp  # rsp can only be a base  # noqa: F405
    assert (m.base, m.index) == (rsp, S)  # noqa: F405
    assert str(qword[S - 8]) == "qword ptr [s-8]"  # noqa: F405
    assert (rip + L).label is L  # noqa: F405
    with pytest.raises(TypeError):
        L + 1


# -- execution ---------------------------------------------------------------


def _sum_fragment() -> Fragment:
    acc, ptr, cnt, done = Hole.gp64("acc"), Hole.gp64("ptr"), Hole.gp64("cnt"), Hole.label("done")
    frag = Fragment(x64)
    with frag:
        mov(acc, 0)  # noqa: F405
        insn_test(cnt, cnt)
        jz(done)  # noqa: F405
        label("loop")
        add(acc, qword[ptr + cnt * 8 - 8])  # noqa: F405
        dec(cnt)  # noqa: F405
        jnz.short("loop")  # noqa: F405
    return frag


@needs_x64_host
def test_exec_instances_with_different_registers():
    frag = _sum_fragment()
    scale = frag_of(lambda: (imul(D, D, I32), add(D, I8)))  # noqa: F405
    a = Assembler(x64)
    with a:
        push(r12)  # noqa: F405
        push(r13)  # noqa: F405
        mov(rcx, rsi)  # noqa: F405
        mov(r12, rsi)  # noqa: F405
        mov(r13, rdi)  # noqa: F405
        d1, d2, d3 = Label(), Label(), Label()
        frag.instantiate(acc=r8, ptr=rdi, cnt=rsi, done=d1)  # noqa: F405
        label(d1)
        frag.instantiate(acc=r9, ptr=r13, cnt=rcx, done=d2)  # noqa: F405
        label(d2)
        frag.instantiate(acc=rax, ptr=r13, cnt=r12, done=d3)  # noqa: F405
        label(d3)
        scale.instantiate(d=r8, i32=1000, i8=-1)  # noqa: F405
        scale.instantiate(d=r9, i32=-3, i8=127)  # noqa: F405
        add(rax, r8)  # noqa: F405
        add(rax, r9)  # noqa: F405
        pop(r13)  # noqa: F405
        pop(r12)  # noqa: F405
        ret()  # noqa: F405
    arr = (ctypes.c_int64 * 4)(1, 2, 3, 4)
    with a.load() as mod:
        f = mod.function(ctypes.c_int64, ctypes.c_void_p, ctypes.c_int64)
        # s + (s*1000 - 1) + (s*-3 + 127) with s = sum
        assert f(ctypes.addressof(arr), 4) == 10 + 9999 + 97
        assert f(ctypes.addressof(arr), 0) == 0 - 1 + 127


@needs_x64_host
def test_exec_all_base_registers():
    # dst = [base + 8] for every base register, including rsp/rbp/r12/r13.
    load = frag_of(lambda: mov(D, qword[S + 8]))  # noqa: F405
    arr = (ctypes.c_int64 * 2)(0, 1234)
    for n in range(16):
        base = gp64(n)  # noqa: F405
        if n == 4:
            continue
        a = Assembler(x64)
        with a:
            push(rbx)  # noqa: F405
            push(rbp)  # noqa: F405
            for r in (r12, r13, r14, r15):  # noqa: F405
                push(r)  # noqa: F405
            mov(base, rdi)  # noqa: F405
            load.instantiate(d=rax, s=base)  # noqa: F405
            for r in (r15, r14, r13, r12):  # noqa: F405
                pop(r)  # noqa: F405
            pop(rbp)  # noqa: F405
            pop(rbx)  # noqa: F405
            ret()  # noqa: F405
        with a.load() as mod:
            f = mod.function(ctypes.c_int64, ctypes.c_void_p)
            assert f(ctypes.addressof(arr)) == 1234, base


def test_extern_slot_inside_fragment_is_created_by_the_instantiating_assembler():
    """`qword[rip + ext]` in a fragment stays a slot request until an
    instance is emitted; each target assembler then gets its own slot."""
    strlen = Extern("strlen")
    p = Hole.gp64("p")
    frag = Fragment(x64)
    with frag:
        mov(rdi, p)
        call(qword[rip + strlen])
    assert list(frag.sections) == ["code"]
    a, b = Assembler(x64), Assembler(x64)
    with a:
        frag.instantiate(p=rbx)
        frag.instantiate(p=rsi)
    with b:
        frag.instantiate(p=rcx)
    for asm in (a, b):
        assert list(asm.extern_slots) == ["strlen"]
        assert list(asm.sections) == ["code", "externs"]
        slot = asm.extern_slots["strlen"]
        assert [(pt.kind, pt.target) for pt in asm.sections["code"].patches if pt.kind is REL32] == [
            (REL32, slot)
        ] * len(asm.sections["code"].insns[1::2])
    img = a.link(0x1000, externs={"strlen": 0x7000})
    slot_addr = img.section_offsets["externs"] + 0x1000
    assert img.data[img.section_offsets["externs"] :][:8] == (0x7000).to_bytes(8, "little")
    # call rel32 at code+0x3 (after `mov rdi, rbx`, 3 bytes) points at the slot
    rel = int.from_bytes(img.data[5:9], "little", signed=True)
    assert 0x1000 + 9 + rel == slot_addr


@needs_x64_host
def test_exec_extern_slot_inside_fragment():
    strlen = Extern("strlen")
    p = Hole.gp64("p")
    frag = Fragment(x64)
    with frag:
        mov(rdi, p)
        call(qword[rip + strlen])
    a = Assembler(x64)
    with a:
        push(rbx)
        mov(rbx, rsi)
        frag.instantiate(p=rdi)
        mov(rsi, rax)
        frag.instantiate(p=rbx)
        add(rax, rsi)
        pop(rbx)
        ret()
    addr = ctypes.cast(ctypes.CDLL(None).strlen, ctypes.c_void_p).value
    with a.load(externs={"strlen": addr}) as mod:
        fn = mod.function(ctypes.c_int64, ctypes.c_char_p, ctypes.c_char_p)
        assert fn(b"abc", b"hello") == 8
