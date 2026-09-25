"""x64 encoder tests.

`CASES` holds byte-exact expectations. Every entry except the rip-relative
ones was produced by running LuaJIT's DynASM (dynasm.lua + dasm_x86.h) on
the same instruction, so these pin jita to DynASM's choice among equivalent
encodings. The oracle tests then check each case against GNU as: either the
bytes are identical, or the case is a known divergence where DynASM picks a
different but equivalent encoding, and objdump must decode our bytes back to
the source instruction.
"""

import ctypes
import platform

import pytest
from oracle import assemble, disassemble, requires_oracle, roundtrip

import jita.x64 as x64
from jita import Assembler, EncodeError, Extern, Label, LinkError
from jita.core import ABS64, REL8, REL32
from jita.x64 import *  # noqa: F403
from jita.x64 import insns
from jita.x64.table import MAP_CC, MAP_OP

# `test` is an x64 mnemonic; keep pytest from collecting it.
insn_test = test  # noqa: F405
del test


def encode(fn, *ops):
    """Encode one instruction into a fresh Assembler, return (bytes, patches)."""
    a = Assembler(x64)
    fn(*ops, asm=a)
    return bytes(a.cur.buf), a.cur.patches


def text_of(fn, ops):
    name = fn.__name__.rstrip("_")
    return name + (" " + ", ".join(str(o) for o in ops) if ops else "")


# fmt: off
CASES = [
    (mov, (rax, rbx), "4889d8"),
    (mov, (eax, ecx), "89c8"),
    (mov, (ax, cx), "6689c8"),
    (mov, (al, cl), "88c8"),
    (mov, (r8, r15), "4d89f8"),
    (mov, (r8d, eax), "4189c0"),
    (mov, (spl, al), "4088c4"),
    (mov, (sil, r9b), "4488ce"),
    (mov, (ah, al), "88c4"),
    (mov, (ah, bh), "88fc"),
    (mov, (rax, qword[rbx]), "488b03"),
    (mov, (qword[rbx], rax), "488903"),
    (mov, (rax, qword[rbx + rcx*8 + 16]), "488b44cb10"),
    (mov, (r8, ptr[r9 + r10*2]), "4f8b0451"),
    (mov, (byte[rax], cl), "8808"),
    (mov, (word[rax], cx), "668908"),
    (mov, (eax, 5), "b805000000"),
    (mov, (r9d, 5), "41b905000000"),
    (mov, (rax, -1), "48c7c0ffffffff"),
    (mov, (rax, 0x7fffffff), "48c7c0ffffff7f"),
    (mov, (r9, 1), "49c7c101000000"),
    (mov, (al, 5), "b005"),
    (mov, (r10b, 200), "41b2c8"),
    (mov, (byte[rax], 5), "c60005"),
    (mov, (word[rax], 0x1234), "66c7003412"),
    (mov, (dword[rax], -1), "c700ffffffff"),
    (mov, (qword[rax], -5), "48c700fbffffff"),
    (mov, (dword[rax], 0xffffffff), "c700ffffffff"),
    (lea, (rax, ptr[rsp + 8]), "488d442408"),
    (lea, (eax, ptr[rax + rbx*2]), "8d0458"),
    (lea, (r12, ptr[r13 + r14*4 - 8]), "4f8d64b5f8"),
    (lea, (rax, ptr[rcx*8]), "488d04cd00000000"),
    (add, (rax, 1), "4883c001"),
    (add, (rcx, 127), "4883c17f"),
    (add, (rcx, 128), "4881c180000000"),
    (add, (rcx, -128), "4883c180"),
    (add, (rcx, -129), "4881c17fffffff"),
    (add, (rax, 128), "480580000000"),
    (add, (eax, 0xffffffff), "83c0ff"),
    (add, (al, 5), "0405"),
    (add, (bl, 200), "80c3c8"),
    (add, (qword[rax], 1), "48830001"),
    (add, (dword[rax], 1000), "8100e8030000"),
    (add, (word[rax], 1000), "668100e803"),
    (add, (r8, r9), "4d01c8"),
    (add, (rax, qword[r8]), "490300"),
    (add, (qword[r8], rax), "490100"),
    (sub, (rsp, 8), "4883ec08"),
    (sub, (rsp, 0x1000), "4881ec00100000"),
    (and_, (rax, rbx), "4821d8"),
    (and_, (ecx, 0xff), "81e1ff000000"),
    (or_, (rax, 1), "4883c801"),
    (or_, (byte[rax], 1), "800801"),
    (xor, (eax, eax), "31c0"),
    (xor, (r15d, r15d), "4531ff"),
    (cmp, (rax, rbx), "4839d8"),
    (cmp, (rcx, 0), "4883f900"),
    (cmp, (byte[rax], 0x7f), "80387f"),
    (cmp, (word[rax], -1), "668338ff"),
    (adc, (rax, 1), "4883d001"),
    (sbb, (ecx, edx), "19d1"),
    (inc, (rax,), "48ffc0"),
    (dec, (dword[rax],), "ff08"),
    (neg, (r8,), "49f7d8"),
    (not_, (ecx,), "f7d1"),
    (mul, (rcx,), "48f7e1"),
    (div, (qword[rax],), "48f730"),
    (idiv, (ecx,), "f7f9"),
    (push, (rax,), "50"),
    (push, (r12,), "4154"),
    (push, (qword[rax],), "ff30"),
    (push, (5,), "6a05"),
    (push, (1000,), "68e8030000"),
    (push, (-1,), "6aff"),
    (push, (ax,), "6650"),
    (pop, (rax,), "58"),
    (pop, (r15,), "415f"),
    (pop, (qword[rbx],), "8f03"),
    (call, (rax,), "ffd0"),
    (call, (r11,), "41ffd3"),
    (call, (qword[rax],), "ff10"),
    (call, (qword[rip + 16],), "ff1510000000"),
    (jmp, (rax,), "ffe0"),
    (jmp, (qword[rax + 8],), "ff6008"),
    (shl, (rax, 1), "48d1e0"),
    (shl, (rax, cl), "48d3e0"),
    (shl, (rax, 5), "48c1e005"),
    (sar, (eax, 3), "c1f803"),
    (ror, (byte[rax], 1), "d008"),
    (shr, (r10d, cl), "41d3ea"),
    (rol, (r9w, 2), "6641c1c102"),
    (sal, (ecx, 1), "d1e1"),
    (imul, (rax, rbx, 10), "486bc30a"),
    (imul, (rax, rbx, 1000), "4869c3e8030000"),
    (imul, (eax, dword[rax], 5), "6b0005"),
    (imul, (r8, r9, -1), "4d6bc1ff"),
    (imul, (rax, rbx), "480fafc3"),
    (imul, (rax, 5), "486bc005"),
    (imul, (rcx,), "48f7e9"),
    (movzx, (eax, cl), "0fb6c1"),
    (movzx, (rax, byte[rbx]), "480fb603"),
    (movzx, (eax, ax), "0fb7c0"),
    (movzx, (ax, cl), "660fb6c1"),
    (movzx, (r8d, sil), "440fb6c6"),
    (movsx, (rax, cl), "480fbec1"),
    (movsx, (eax, word[rax]), "0fbf00"),
    (movsx, (r9, r10w), "4d0fbfca"),
    (movsxd, (rax, ecx), "4863c1"),
    (movsxd, (r8, dword[rbx]), "4c6303"),
    (insn_test, (rax, rbx), "4885d8"),
    (insn_test, (al, 1), "a801"),
    (insn_test, (eax, 0x100), "a900010000"),
    (insn_test, (ecx, 5), "f7c105000000"),
    (insn_test, (qword[rax], 7), "48f70007000000"),
    (insn_test, (byte[rax], 1), "f60001"),
    (xchg, (rax, rbx), "4893"),
    (xchg, (rcx, rax), "4891"),
    (xchg, (rcx, rdx), "4887ca"),
    (bswap, (eax,), "0fc8"),
    (bswap, (r9,), "490fc9"),
    (bt, (rax, 3), "480fbae003"),
    (bsf, (rax, rcx), "480fbcc1"),
    (popcnt, (rax, rcx), "f3480fb8c1"),
    (lzcnt, (eax, ecx), "f30fbdc1"),
    (tzcnt, (r8, qword[rax]), "f34c0fbc00"),
    (cmovz, (rax, rbx), "480f44c3"),
    (cmovl, (r8d, dword[rax]), "440f4c00"),
    (cmovnbe, (ax, cx), "660f47c1"),
    (sete, (al,), "0f94d0"),
    (setne, (byte[rax],), "0f9510"),
    (setg, (r8b,), "410f9fd0"),
    (setb, (sil,), "400f92d6"),
    (ret, (), "c3"),
    (ret, (16,), "c21000"),
    (nop, (), "90"),
    (int3, (), "cc"),
    (int_, (3,), "cd03"),
    (cqo, (), "4899"),
    (cdqe, (), "4898"),
    (cdq, (), "99"),
    (leave, (), "c9"),
    (mov, (rax, qword[rsp]), "488b0424"),
    (mov, (rax, qword[rsp + 8]), "488b442408"),
    (mov, (rax, qword[rbp]), "488b4500"),
    (mov, (rax, qword[r13]), "498b4500"),
    (mov, (rax, qword[r12]), "498b0424"),
    (mov, (rax, qword[r12 + 8]), "498b442408"),
    (mov, (rax, qword[r12 + 0x1000]), "498b842400100000"),
    (mov, (rax, qword[rcx*8 + 16]), "488b04cd10000000"),
    (mov, (rax, qword[0x1000]), "488b042500100000"),
    (mov, (rax, qword[rbp + rcx*2]), "488b444d00"),
    (mov, (rax, qword[r13 + r12*1]), "4b8b442500"),
    (mov, (rax, qword[rax + rsp]), "488b0404"),
    (mov, (rax, qword[rbx + 127]), "488b437f"),
    (mov, (rax, qword[rbx + 128]), "488b8380000000"),
    (mov, (rax, qword[rbx - 128]), "488b4380"),
    (mov, (rax, qword[rbx - 129]), "488b837fffffff"),
    (movsd, (xmm0, xmm1), "f20f10c1"),
    (movsd, (xmm0, qword[rax]), "f20f1000"),
    (movsd, (qword[rax], xmm1), "f20f1108"),
    (movsd, (xmm8, qword[r9 + 8]), "f2450f104108"),
    (addsd, (xmm8, xmm1), "f2440f58c1"),
    (addsd, (xmm0, qword[rip + 8]), "f20f580508000000"),
    (subsd, (xmm1, xmm2), "f20f5cca"),
    (mulsd, (xmm1, xmm2), "f20f59ca"),
    (divsd, (xmm1, xmm2), "f20f5eca"),
    (sqrtsd, (xmm1, xmm2), "f20f51ca"),
    (addss, (xmm0, dword[rax]), "f30f5800"),
    (addps, (xmm0, xmm1), "0f58c1"),
    (movaps, (xmm0, xmm1), "0f28c1"),
    (movaps, (oword[rax], xmm15), "440f2938"),
    (movaps, (xmm3, oword[rsp + 16]), "0f285c2410"),
    (movups, (xmm0, oword[rax]), "0f1000"),
    (movapd, (xmm1, xmm2), "660f28ca"),
    (movdqa, (xmm0, oword[rax]), "660f6f00"),
    (movdqu, (oword[rax], xmm9), "f3440f7f08"),
    (cvtsi2sd, (xmm0, rax), "f2480f2ac0"),
    (cvtsi2sd, (xmm0, eax), "f20f2ac0"),
    (cvtsi2sd, (xmm0, dword[rax]), "f20f2a00"),
    (cvtsi2sd, (xmm0, qword[rax]), "f2480f2a00"),
    (cvttsd2si, (rax, xmm0), "f2480f2cc0"),
    (cvttsd2si, (eax, xmm1), "f20f2cc1"),
    (cvtsd2ss, (xmm0, xmm1), "f20f5ac1"),
    (cvtss2sd, (xmm0, dword[rax]), "f30f5a00"),
    (pxor, (xmm0, xmm0), "660fefc0"),
    (pxor, (xmm9, xmm10), "66450fefca"),
    (paddd, (xmm1, oword[rax]), "660ffe08"),
    (pand, (xmm1, xmm2), "660fdbca"),
    (movq, (xmm0, rax), "66480f6ec0"),
    (movq, (rax, xmm0), "66480f7ec0"),
    (movq, (xmm0, xmm1), "f30f7ec1"),
    (movq, (xmm0, qword[rax]), "f30f7e00"),
    (movq, (qword[rax], xmm0), "660fd600"),
    (movd, (xmm0, eax), "660f6ec0"),
    (movd, (eax, xmm1), "660f7ec8"),
    (ucomisd, (xmm0, xmm1), "660f2ec1"),
    (comiss, (xmm0, dword[rax]), "0f2f00"),
    (xorps, (xmm0, xmm0), "0f57c0"),
    (andpd, (xmm0, xmm1), "660f54c1"),
    (pshufd, (xmm0, xmm1, 0x1b), "660f70c11b"),
    (shufps, (xmm0, xmm1, 0x44), "0fc6c144"),
    (pextrd, (eax, xmm1, 2), "660f3a16c802"),
    (pinsrq, (xmm1, rax, 1), "66480f3a22c801"),
    (pslld, (xmm0, 4), "660f72f004"),
    (psrlq, (xmm0, xmm1), "660fd3c1"),
    (roundsd, (xmm0, xmm1, 4), "660f3a0bc104"),
    (blendvps, (xmm1, xmm2, xmm0), "660f3814ca"),
    (vaddps, (ymm1, ymm2, ymm3), "c5ec58cb"),
    (vaddps, (xmm1, xmm2, oword[rax]), "c5e85808"),
    (vaddps, (ymm8, ymm9, ymm10), "c4413458c2"),
    (vaddsd, (xmm0, xmm1, xmm2), "c5f358c2"),
    (vaddsd, (xmm0, xmm1, qword[rax]), "c5f35800"),
    (vmulpd, (ymm0, ymm1, yword[rax + 32]), "c5f5594020"),
    (vmovups, (ymm0, yword[rax]), "c5fc1000"),
    (vmovups, (yword[rax], ymm8), "c57c1100"),
    (vmovups, (yword[r8], ymm0), "c4c17c1100"),
    (vmovups, (xmm0, xmm1), "c5f810c1"),
    (vmovaps, (ymm1, ymm2), "c5fc28ca"),
    (vmovdqu, (ymm0, yword[rax]), "c5fe6f00"),
    (vpxor, (ymm0, ymm1, ymm15), "c4c175efc7"),
    (vpxor, (xmm0, xmm1, xmm2), "c5f1efc2"),
    (vxorps, (xmm0, xmm0, xmm0), "c5f857c0"),
    (vfmadd231pd, (ymm0, ymm1, ymm2), "c4e2f5b8c2"),
    (vfmadd231sd, (xmm0, xmm1, qword[rax]), "c4e2f1b900"),
    (vblendvps, (xmm0, xmm1, xmm2, xmm3), "c4e3714ac230"),
    (vshufps, (ymm0, ymm1, ymm2, 0x1b), "c5f4c6c21b"),
    (vcvtsi2sd, (xmm0, xmm1, rax), "c4e1f32ac0"),
    (vbroadcastss, (ymm0, dword[rax]), "c4e27d1800"),
    (vpermq, (ymm0, ymm1, 0x4e), "c4e3fd00c14e"),
    (vextractf128, (xmm0, ymm1, 1), "c4e37d19c801"),
    (vinsertf128, (ymm0, ymm1, oword[rax], 1), "c4e375180001"),
    (vpsllq, (ymm0, ymm1, 3), "c5fd73f103"),
    (vzeroupper, (), "c5f877"),
    (vzeroall, (), "c5fc77"),
    (andn, (rax, rbx, rcx), "c4e2e0f2c1"),
    (bextr, (eax, ecx, edx), "c4e268f7c1"),
    (shlx, (rax, qword[rbx], rcx), "c4e2f1f703"),
    (rorx, (rax, rbx, 5), "c4e3fbf0c305"),
    (crc32, (eax, cl), "f20f38f0c1"),
    (crc32, (rax, qword[rbx]), "f2480f38f103"),
]
# fmt: on

# DynASM picks a different (equivalent) encoding than GNU as for these.
# The value is how objdump prints our bytes, or None if it prints the source.
AS_DIVERGENT = {
    "xchg rcx, rdx": "xchg rdx,rcx",  # DynASM: 87 /r with rcx in reg field
    "int 3": None,  # as: CC; DynASM: CD 03
}


def _case_id(case):
    fn, ops, _ = case
    return text_of(fn, ops)


@pytest.mark.parametrize("case", CASES, ids=_case_id)
def test_bytes(case):
    fn, ops, expected = case
    code, patches = encode(fn, *ops)
    assert code.hex() == expected
    assert patches == []


@requires_oracle
@pytest.mark.parametrize("case", CASES, ids=_case_id)
def test_against_gas(case):
    fn, ops, expected = case
    text = text_of(fn, ops)
    code = bytes.fromhex(expected)
    if text.startswith("set"):
        # DynASM fills the unused ModRM reg field of SETcc with 2 ("0F9x2m").
        roundtrip(text, code)
    elif text in AS_DIVERGENT:
        alt = AS_DIVERGENT[text]
        if alt is None:
            roundtrip(text, code)
        else:
            assert disassemble(code) == [alt]
    else:
        assert assemble(text) == code


# -- generated families ------------------------------------------------------


@pytest.mark.parametrize("cc", sorted(MAP_CC))
def test_jcc_long_and_short(cc):
    n = MAP_CC[cc]
    fn = getattr(x64, "j" + cc)
    lbl = Label()
    code, patches = encode(fn, lbl)
    assert code == bytes([0x0F, 0x80 + n, 0, 0, 0, 0])
    assert [(p.offset, p.kind, p.target) for p in patches] == [(2, REL32, lbl)]
    code, patches = encode(fn.short, lbl)
    assert code == bytes([0x70 + n, 0])
    assert [(p.offset, p.kind, p.target) for p in patches] == [(1, REL8, lbl)]


@pytest.mark.parametrize("cc", sorted(MAP_CC))
def test_setcc_cmovcc(cc):
    n = MAP_CC[cc]
    assert encode(getattr(x64, "set" + cc), al)[0] == bytes([0x0F, 0x90 + n, 0xD0])
    assert encode(getattr(x64, "set" + cc), byte[r9])[0] == bytes([0x41, 0x0F, 0x90 + n, 0x11])
    assert encode(getattr(x64, "cmov" + cc), rax, rbx)[0] == bytes([0x48, 0x0F, 0x40 + n, 0xC3])
    assert encode(getattr(x64, "cmov" + cc), r8d, dword[rax])[0] == bytes([0x44, 0x0F, 0x40 + n, 0x00])


@requires_oracle
def test_jcc_linked_against_gas():
    a = Assembler(x64)
    src = []
    for cc in sorted(MAP_CC):
        top, fwd = a.label(), Label()
        getattr(a, "j" + cc)(fwd)
        getattr(a, "j" + cc).short(fwd)
        a.nop()
        a.bind(fwd)
        getattr(a, "j" + cc)(top)
        getattr(a, "j" + cc).short(top)
        src += ["0:", f"{{disp32}} j{cc} 1f", f"j{cc} 1f", "nop", "1:", f"{{disp32}} j{cc} 0b", f"j{cc} 0b"]
    img = a.link()
    assert img.data == assemble("\n".join(src))


# -- calls, jumps and labels ---------------------------------------------------


def test_jmp_call_label():
    lbl = Label()
    code, patches = encode(jmp, lbl)
    assert code == b"\xe9\0\0\0\0" and [(p.offset, p.kind) for p in patches] == [(1, REL32)]
    code, patches = encode(jmp.short, lbl)
    assert code == b"\xeb\0" and [(p.offset, p.kind) for p in patches] == [(1, REL8)]
    code, patches = encode(call, lbl)
    assert code == b"\xe8\0\0\0\0" and [(p.offset, p.kind) for p in patches] == [(1, REL32)]


def test_call_extern_rel32():
    ext = Extern("memcpy")
    code, patches = encode(call, ext)
    assert code == b"\xe8\0\0\0\0"
    assert patches[0].target is ext and patches[0].kind is REL32


def test_short_needs_label():
    with pytest.raises(EncodeError, match="short"):
        encode(jmp.short, rax)
    assert not hasattr(call, "short")


def test_short_out_of_range_fails_at_link():
    a = Assembler(x64)
    lbl = Label()
    a.jmp.short(lbl)
    a.space(200)
    a.bind(lbl)
    with pytest.raises(LinkError):
        a.link()


def test_linked_jumps():
    a = Assembler(x64)
    top, end = Label(), Label()
    a.bind(top)
    a.jmp(end)
    a.call(top)
    a.jz.short(end)
    a.bind(end)
    a.jmp.short(top)
    img = a.link()
    assert img.data == bytes.fromhex("e907000000" "e8f6ffffff" "7400" "ebf2")


# -- rip-relative -----------------------------------------------------------


def test_rip_relative_label():
    lbl = Label()
    code, patches = encode(mov, rax, qword[rip + lbl])
    assert code == bytes.fromhex("488b0500000000")
    assert [(p.offset, p.kind, p.target, p.addend) for p in patches] == [(3, REL32, lbl, 0)]
    # disp is the addend
    _, patches = encode(lea, rax, ptr[rip + lbl + 8])
    assert patches[0].addend == 8


def test_rip_relative_followed_by_immediate():
    # rip points past the immediate, so the addend is corrected by its size.
    lbl = Label()
    code, patches = encode(mov, dword[rip + lbl], 5)
    assert code == bytes.fromhex("c7050000000005000000")
    assert (patches[0].offset, patches[0].addend) == (2, -4)
    code, patches = encode(cmp, byte[rip + lbl], 1)
    assert code == bytes.fromhex("803d0000000001")
    assert (patches[0].offset, patches[0].addend) == (2, -1)
    code, patches = encode(pshufd, xmm0, oword[rip + lbl + 16], 0x1B)
    assert code == bytes.fromhex("660f7005000000001b")
    assert (patches[0].offset, patches[0].addend) == (4, 16 - 1)


@requires_oracle
def test_rip_relative_linked_against_gas():
    a = Assembler(x64)
    lbl = Label()
    a.mov(rax, qword[rip + lbl])
    a.mov(dword[rip + lbl], 5)
    a.add(qword[rip + lbl + 8], 1000)
    a.movsd(xmm9, qword[rip + lbl])
    a.vmovups(ymm0, yword[rip + lbl])
    a.bind(lbl)
    a.nop()
    src = """
        mov rax, qword ptr [rip+1f]
        mov dword ptr [rip+1f], 5
        add qword ptr [rip+1f+8], 1000
        movsd xmm9, qword ptr [rip+1f]
        vmovups ymm0, ymmword ptr [rip+1f]
        1: nop
    """
    assert a.link().data == assemble(src)


# -- mov with 64 bit immediates ------------------------------------------------


def test_mov_imm64():
    assert encode(mov, rax, 0x123456789ABCDEF0)[0].hex() == "48b8f0debc9a78563412"
    assert encode(mov, r9, 0x80000000)[0].hex() == "49b90000008000000000"
    assert encode(mov, rax, -(1 << 63))[0].hex() == "48b80000000000000080"
    assert encode(mov, rax, 0xFFFFFFFFFFFFFFFF)[0].hex() == "48b8ffffffffffffffff"
    # Values that fit a sign-extended imm32 keep DynASM's C7 form.
    assert encode(mov, rax, -0x80000000)[0].hex() == "48c7c000000080"
    assert encode(mov64, rax, 5)[0].hex() == "48b80500000000000000"
    assert encode(movabs, r15, 5)[0].hex() == "49bf0500000000000000"


def test_mov_label_address():
    lbl = Label()
    code, patches = encode(mov, rcx, lbl)
    assert code.hex() == "48b90000000000000000"
    assert [(p.offset, p.kind, p.target) for p in patches] == [(2, ABS64, lbl)]
    ext = Extern("puts")
    code, patches = encode(mov64, r10, ext)
    assert code.hex() == "49ba0000000000000000" and patches[0].target is ext


def test_mov64_moffs():
    assert encode(mov64, rax, qword[0x1000])[0].hex() == "48a10010000000000000"
    assert encode(mov64, dword[0x1000], eax)[0].hex() == "a30010000000000000"
    assert encode(mov64, al, byte[0x10])[0].hex() == "a01000000000000000"
    assert encode(mov64, word[0x10], ax)[0].hex() == "66a31000000000000000"
    with pytest.raises(EncodeError):
        encode(mov64, rcx, qword[0x1000])


@requires_oracle
def test_mov_imm64_against_gas():
    for text, ops in [
        ("movabs rax, 0x123456789abcdef0", (rax, 0x123456789ABCDEF0)),
        ("movabs r9, 0x80000000", (r9, 0x80000000)),
    ]:
        assert encode(mov, *ops)[0] == assemble(text)
    assert encode(mov64, rax, qword[0x1000])[0] == assemble("movabs rax, [0x1000]")
    assert encode(mov64, dword[0x1000], eax)[0] == assemble("movabs [0x1000], eax")


# -- address size and misc encoder behavior -----------------------------------


@requires_oracle
def test_addr32_prefix():
    code = encode(mov, eax, dword[ecx + edx * 4 + 8])[0]
    assert code.hex() == "678b449108"
    assert code == assemble("mov eax, dword ptr [ecx+edx*4+8]")
    code = encode(vaddps, xmm0, xmm1, oword[r8d])[0]
    assert code == assemble("vaddps xmm0, xmm1, xmmword ptr [r8d]")


def test_imm_ranges():
    assert encode(mov, al, -1)[0].hex() == "b0ff"  # DynASM rejects, jita masks
    assert encode(add, ax, -129)[0].hex() == "66057fff"
    assert encode(cmp, byte[rax], 255)[0].hex() == "8038ff"
    for fn, ops in [
        (add, (rax, 0xFFFFFFFF)),  # DynASM: silently add rax, -1
        (add, (rax, 1 << 32)),
        (add, (al, 256)),
        (add, (ax, -32769)),
        (add, (ecx, 1 << 32)),
        (shl, (eax, 256)),
        (shl, (eax, -1)),
        (ret, (0x10000,)),
        (imul, (rax, rbx, 0x80000000)),
        (push, (0xFFFFFFFF,)),  # DynASM: push -1
        (push, (0x80000000,)),
    ]:
        with pytest.raises(EncodeError, match="range"):
            encode(fn, *ops)


def test_high_byte_registers():
    assert encode(mov, ah, al)[0].hex() == "88c4"
    assert encode(mov, ah, byte[rax])[0].hex() == "8a20"
    assert encode(movzx, eax, ah)[0].hex() == "0fb6c4"
    for ops in [(ah, sil), (ah, r8b), (ah, byte[r8]), (ah, byte[rax + r9])]:
        with pytest.raises(EncodeError):
            encode(mov, *ops)
    with pytest.raises(EncodeError, match="REX"):
        encode(movzx, rax, ah)
    with pytest.raises(EncodeError, match="REX"):
        encode(movzx, r8d, bh)


def test_error_messages():
    with pytest.raises(EncodeError) as e:
        encode(mov, qword[rax], qword[rbx])
    assert str(e.value).startswith("mov: no encoding for (qword ptr [rax], qword ptr [rbx])")
    with pytest.raises(EncodeError, match="missing operand size"):
        encode(mov, ptr[rax], 1)
    with pytest.raises(EncodeError, match="mixed operand size"):
        encode(mov, rax, ecx)
    with pytest.raises(EncodeError, match="bad operand size"):
        encode(push, eax)
    with pytest.raises(EncodeError, match="expects 2 operands"):
        encode(mov, rax)
    with pytest.raises(EncodeError, match="and: no encoding"):
        encode(and_, rax, xmm0)
    with pytest.raises(EncodeError):
        encode(push, Label())  # DynASM would emit a broken 8 byte immediate
    with pytest.raises(EncodeError):
        encode(mov, rax, rip)
    with pytest.raises(EncodeError, match="VSIB"):
        encode(mov, rax, qword[rax + xmm1 * 2])


def test_failed_encoding_emits_nothing():
    a = Assembler(x64)
    a.nop()
    with pytest.raises(EncodeError):
        a.add(rax, 1 << 40)
    assert bytes(a.cur.buf) == b"\x90"


# -- mnemonic functions -------------------------------------------------------


def test_insns_namespace():
    names = {k.rpartition("_")[0] for k in MAP_OP}
    for name in names:
        py = insns.PY_NAMES.get(name, name)
        assert py in insns.INSNS, name
    assert {"and_", "or_", "not_", "int_", "mov64", "movabs"} <= set(insns.INSNS)
    assert "and" not in insns.__all__ and "int" not in insns.__all__
    assert x64.and_ is insns.INSNS["and_"]
    assert jz.short.__qualname__ == "jz.short" and jmp.short
    assert not hasattr(mov, "short")
    assert "in" not in insns.PY_NAMES and "in_" not in insns.INSNS
    assert x64.ARCH.insns is insns.INSNS


def test_current_assembler_and_methods():
    a = Assembler(x64)
    with a:
        mov(rax, rbx)
        and_(eax, 1)
    a.or_(rax, rcx)
    a.jmp.short(a.label())
    assert bytes(a.cur.buf).hex() == "4889d8" "83e001" "4809c8" "eb00"


def test_table_size():
    assert len(MAP_OP) == 941 + 11  # DynASM + jita extensions


# -- execution ---------------------------------------------------------------

needs_x64_host = pytest.mark.skipif(
    platform.machine().lower() not in ("x86_64", "amd64") or platform.system() == "Windows",
    reason="needs an x86-64 SysV host",
)


@needs_x64_host
def test_exec_sum_array():
    a = Assembler(x64)
    with a:
        loop, done = Label(), Label()
        xor(eax, eax)
        insn_test(rsi, rsi)
        jz.short(done)
        a.bind(loop)
        add(rax, qword[rdi + rsi * 8 - 8])
        dec(rsi)
        jnz(loop)
        a.bind(done)
        ret()
    arr = (ctypes.c_int64 * 5)(1, 2, 3, 4, 1000)
    with a.load() as mod:
        f = mod.function(ctypes.c_int64, ctypes.c_void_p, ctypes.c_int64)
        assert f(ctypes.addressof(arr), 5) == 1010
        assert f(ctypes.addressof(arr), 0) == 0


@needs_x64_host
def test_exec_sse_and_data():
    a = Assembler(x64)
    with a:
        with a.section("data"):
            a.align(8)
            k = a.label()
            a.qword(0x4000000000000000)  # 2.0
        cvtsi2sd(xmm1, rdi)
        mulsd(xmm1, qword[rip + k])
        movapd(xmm0, xmm1)
        ret()
    with a.load() as mod:
        assert mod.function(ctypes.c_double, ctypes.c_int64)(21) == 42.0


@needs_x64_host
def test_exec_call_extern_and_movabs():
    cb = ctypes.CFUNCTYPE(ctypes.c_int64, ctypes.c_int64)(lambda v: v + 1)
    addr = ctypes.cast(cb, ctypes.c_void_p).value
    a = Assembler(x64)
    with a:
        push(rbx)
        mov(rbx, 0x100000000)
        mov(rax, Extern("inc"))
        call(rax)
        add(rax, rbx)
        pop(rbx)
        ret()
    with a.load(externs={"inc": addr}) as mod:
        assert mod.function(ctypes.c_int64, ctypes.c_int64)(41) == 0x100000000 + 42


# -- x87 stack registers -------------------------------------------------------

# (fn, ops, bytes); bytes checked against DynASM templates and GNU as.
X87_CASES = [
    (fadd, (st0, st1), "d8c1"),
    (fadd, (st3, st0), "dcc3"),
    (fadd, (st2,), "d8c2"),
    (fmul, (st0, st7), "d8cf"),
    (fdivr, (st4, st0), "dcf4"),
    (faddp, (st1, st0), "dec1"),
    (fsubp, (st2,), "deea"),
    (fld, (st1,), "d9c1"),
    (fst, (st2,), "ddd2"),
    (fstp, (st0,), "ddd8"),
    (fxch, (st1,), "d9c9"),
    (fcomi, (st0, st1), "dbf1"),
    (fucomip, (st0, st2), "dfea"),
    (fcmovb, (st0, st3), "dac3"),
]


@pytest.mark.parametrize("case", X87_CASES, ids=_case_id)
def test_x87_bytes(case):
    fn, ops, expected = case
    assert encode(fn, *ops)[0].hex() == expected


@requires_oracle
@pytest.mark.parametrize("case", X87_CASES, ids=_case_id)
def test_x87_against_gas(case):
    fn, ops, expected = case
    assert assemble(text_of(fn, ops)) == bytes.fromhex(expected)


def test_x87_two_operand_fxch():
    # DynASM accepts `fxch st0, stN` and `fxch stN, st0`; GNU as does not.
    assert encode(fxch, st0, st5)[0].hex() == "d9cd"
    assert encode(fxch, st5, st0)[0].hex() == "d9cd"
    assert st(3) is st3 and str(st3) == "st(3)"


def test_x87_errors():
    with pytest.raises(EncodeError):
        encode(fadd, st1, st2)  # one side must be st0
    with pytest.raises(EncodeError):
        encode(mov, rax, st0)
    with pytest.raises(EncodeError):
        st1 + 8  # not an address register


# -- review fixes ----------------------------------------------------------------


def test_xchg_eax_eax_is_not_nop():
    assert encode(xchg, eax, eax)[0].hex() == "87c0"
    # Other short forms are unchanged.
    assert encode(xchg, rax, rax)[0].hex() == "4890"
    assert encode(xchg, ax, ax)[0].hex() == "6690"
    assert encode(xchg, eax, ecx)[0].hex() == "91"


@requires_oracle
def test_xchg_eax_eax_against_gas():
    assert encode(xchg, eax, eax)[0] == assemble("xchg eax, eax")


@needs_x64_host
def test_exec_xchg_eax_eax_zero_extends():
    a = Assembler(x64)
    with a:
        mov(rax, -1)
        xchg(eax, eax)
        ret()
    with a.load() as mod:
        assert mod.function(ctypes.c_uint64)() == 0xFFFFFFFF


def test_absolute_64bit_address():
    assert encode(mov64, rax, qword[0x100000000])[0].hex() == "48a10000000001000000"
    assert encode(mov64, dword[0x100000000], eax)[0].hex() == "a30000000001000000"
    for fn, ops in [
        (mov, (rax, qword[0x100000000])),
        (add, (qword[1 << 40], 1)),
        (lea, (rax, ptr[1 << 32])),
    ]:
        with pytest.raises(EncodeError, match="use mov64"):
            encode(fn, *ops)
    # A negative absolute address still fits the sign-extended disp32 form.
    assert encode(mov, rax, qword[-8])[0].hex() == "488b0425f8ffffff"


@requires_oracle
def test_absolute_64bit_address_against_gas():
    text = "movabs rax, qword ptr [0x100000000]"
    assert encode(mov64, rax, qword[0x100000000])[0] == assemble(text)
    roundtrip("movabs rax, ds:0x100000000", encode(movabs, rax, qword[0x100000000])[0])


def test_label_immediate_error_message():
    for fn, ops in [(mov, (eax, Label())), (add, (rax, Label())), (push, (Label(),))]:
        with pytest.raises(EncodeError, match=r"only valid as .* source of mov r64 / mov64"):
            encode(fn, *ops)


# (fn, ops, GNU as text) for the jita extensions in table.py.
JITA_EXTENSIONS = [
    (xadd, (qword[rax], rbx), "xadd qword ptr [rax], rbx"),
    (xadd, (dword[rax + rcx * 4], r9d), "xadd dword ptr [rax+rcx*4], r9d"),
    (xadd, (word[r9 + 8], r10w), "xadd word ptr [r9+8], r10w"),
    (xadd, (byte[rax], sil), "xadd byte ptr [rax], sil"),
    (xadd, (ecx, edx), "xadd ecx, edx"),
    (cmpxchg, (qword[rdi], rsi), "cmpxchg qword ptr [rdi], rsi"),
    (cmpxchg, (dword[rdi], r8d), "cmpxchg dword ptr [rdi], r8d"),
    (cmpxchg, (word[rdi], cx), "cmpxchg word ptr [rdi], cx"),
    (cmpxchg, (byte[rdi], cl), "cmpxchg byte ptr [rdi], cl"),
    (cmpxchg, (r12, rax), "cmpxchg r12, rax"),
    (cmpxchg8b, (qword[rdi],), "cmpxchg8b qword ptr [rdi]"),
    (cmpxchg16b, (oword[r8],), "cmpxchg16b xmmword ptr [r8]"),
    (ud2, (), "ud2"),
    (hlt, (), "hlt"),
    (movsq, (), "movsq"),
    (cmpsq, (), "cmpsq"),
    (stosq, (), "stosq"),
    (lodsq, (), "lodsq"),
    (scasq, (), "scasq"),
]


@requires_oracle
@pytest.mark.parametrize("case", JITA_EXTENSIONS, ids=lambda c: c[2])
def test_jita_extensions_against_gas(case):
    fn, ops, text = case
    code = encode(fn, *ops)[0]
    assert code == assemble(text)
    if ops:  # objdump prints string ops with implicit operands
        roundtrip(text.replace("xmmword", "oword"), code)


@requires_oracle
def test_prefixed_string_and_atomic_ops_against_gas():
    a = Assembler(x64)
    with a:
        lock()
        cmpxchg(qword[rdi], rsi)
        lock()
        xadd(dword[rdi], eax)
        rep()
        movsq()
        rep()
        stosb()
        repe()
        cmpsq()
        repne()
        scasb()
    text = """
        lock cmpxchg qword ptr [rdi], rsi
        lock xadd dword ptr [rdi], eax
        rep movsq
        rep stosb
        repe cmpsq
        repne scasb
    """
    assert bytes(a.cur.buf) == assemble(text)


@needs_x64_host
def test_exec_lock_xadd_and_cmpxchg():
    a = Assembler(x64)
    with a:
        # rdi -> int64 counter; returns the old value, adds 5, then CAS 5 -> 7.
        mov(eax, 5)
        lock()
        xadd(qword[rdi], rax)
        mov(rcx, rax)
        mov(eax, 5)
        mov(edx, 7)
        lock()
        cmpxchg(qword[rdi], rdx)
        mov(rax, rcx)
        ret()
    v = ctypes.c_int64(0)
    with a.load() as mod:
        assert mod.function(ctypes.c_int64, ctypes.c_void_p)(ctypes.addressof(v)) == 0
    assert v.value == 7


@needs_x64_host
def test_exec_rep_movsq():
    a = Assembler(x64)
    with a:
        # memcpy(rdi, rsi, rdx qwords)
        mov(rcx, rdx)
        rep()
        movsq()
        ret()
    src = (ctypes.c_int64 * 4)(1, 2, 3, 4)
    dst = (ctypes.c_int64 * 4)()
    with a.load() as mod:
        f = mod.function(None, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64)
        f(ctypes.addressof(dst), ctypes.addressof(src), 4)
    assert list(dst) == [1, 2, 3, 4]


def test_bound_insn_attributes():
    a = Assembler(x64)
    assert a.mov.__doc__ == mov.__doc__ and "mov" in a.mov.__doc__
    assert a.mov.__name__ == "mov"
    assert a.jz.short.__doc__ == jz.short.__doc__
    a.jz.short(a.label())
    assert bytes(a.cur.buf).hex() == "7400"


def test_star_export_is_limited():
    ns: dict = {}
    exec("from jita.x64 import *", ns)
    for leaked in ("annotations", "Any", "Callable", "Arch", "mem", "regs", "insns", "table", "encoder"):
        assert leaked not in ns, leaked
    for name in ("ARCH", "mov", "and_", "rax", "ah", "st0", "qword", "ptr", "MemExpr", "gp64"):
        assert name in ns, name
    assert "int" not in ns and "int_" in ns


def test_absolute_address_unsigned_spelling():
    # 0xffffffff80000000 is the sign-extended disp32 -0x80000000.
    assert encode(mov, rax, qword[0xFFFFFFFF80000000])[0] == encode(mov, rax, qword[-0x80000000])[0]
    assert encode(mov, rax, qword[0xFFFFFFFFFFFFFFF8])[0].hex() == "488b0425f8ffffff"
    with pytest.raises(EncodeError, match="use mov64"):
        encode(mov, rax, qword[0xFFFFFFFF7FFFFFFF])
    with pytest.raises(EncodeError, match="use mov64"):
        encode(mov, rax, qword[0x80000000])


@requires_oracle
def test_absolute_address_unsigned_spelling_against_gas():
    code = encode(mov, rax, qword[0xFFFFFFFF80000000])[0]
    assert code == assemble("mov rax, qword ptr [0xffffffff80000000]")


@needs_x64_host
def test_exec_rip_relative_with_immediates():
    # Each store has a different immediate size after the rip-relative disp.
    a = Assembler(x64)
    with a:
        d = Label()
        mov(dword[rip + d], 0x12345678)
        mov(word[rip + d + 4], 0x9ABC)
        mov(byte[rip + d + 6], 0xDE)
        add(byte[rip + d + 7], 1)
        mov(rax, qword[rip + d])
        ret()
        with a.section("data", writable=True):
            label(d)
            a.qword(0)
    with a.load() as mod:
        assert mod.function(ctypes.c_uint64)() == 0x01DE9ABC12345678


def test_string_label_operands():
    """A str operand names a label of the assembler, DynASM's `->name`."""
    a = Assembler(x64)
    with a:
        jz("done")
        jmp.short("done")
        mov(rax, "tbl")
        lea(rcx, ptr[rip + "tbl"])
        mov(rdx, qword[rip + "tbl" + 8])
        call("done")
        a.label("done")
        ret()
        with a.section("data"):
            a.label("tbl")
            a.qword("done", 1)
    code = a.sections["code"]
    kinds = [(p.kind, p.target.name) for p in code.patches]
    assert kinds == [
        (REL32, "done"), (REL8, "done"), (ABS64, "tbl"), (REL32, "tbl"),
        (REL32, "tbl"), (REL32, "done"),
    ]
    img = a.link(0x1000)
    done, tbl = img.address("done"), img.address("tbl")
    b = img.data
    assert int.from_bytes(b[2:6], "little", signed=True) == done - 0x1006
    assert b[7] == done - 0x1008
    assert int.from_bytes(b[10:18], "little") == tbl
    data = img.section_offsets["data"]
    assert int.from_bytes(b[data : data + 8], "little") == done  # a.qword("done")


def test_string_label_identity_with_label_object():
    a = Assembler(x64)
    with a:
        jmp("loop")
        loop = a.named("loop")
        assert loop is a.named("loop")
        a.label("loop")
        jmp(loop)
    assert [p.target for p in a.sections["code"].patches] == [loop, loop]


def test_string_label_error_messages():
    a = Assembler(x64)
    with a:
        jmp("x")
    with pytest.raises(LinkError, match="label x is never bound"):
        a.link()
    with pytest.raises(EncodeError, match=r"use qword\[rip \+ label\]"):
        qword["x"]


@requires_oracle
def test_string_label_matches_object_label_bytes():
    a, b = Assembler(x64), Assembler(x64)
    with a:
        jz("end"); add(rax, qword[rip + "k"]); jmp.short("end")
        a.label("k"); a.qword(0)
        a.label("end"); ret()
    with b:
        end, k = Label("end"), Label("k")
        jz(end); add(rax, qword[rip + k]); jmp.short(end)
        label(k); b.qword(0)
        label(end); ret()
    assert a.link().data == b.link().data


@needs_x64_host
def test_exec_string_labels():
    a = Assembler(x64)
    with a:
        xor(eax, eax)
        x64.test(rsi, rsi)
        jz("done")
        a.label("loop")
        add(rax, qword[rdi])
        add(rdi, 8)
        dec(rsi)
        jnz("loop")
        a.label("done")
        ret()
    with a.load() as mod:
        fn = mod.function(ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_size_t)
        arr = (ctypes.c_int64 * 4)(1, 2, 3, 4)
        assert fn(arr, 4) == 10
        assert fn(arr, 0) == 0
