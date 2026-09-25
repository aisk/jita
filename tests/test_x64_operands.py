import pytest

from jita import Label
from jita.core import EncodeError, Imm, Register
from jita.x64 import (
    MemExpr, ah, al, byte, dword, eax, ecx, gp8, gp32, gp64, oword, ptr, qword,
    r8b, r12, r13d, r15, rax, rbp, rbx, rcx, rip, rsp, sil, spl, word, xmm, xmm3, ymm, yword,
)


def test_register_attributes():
    assert isinstance(rax, Register)
    assert (rax.kind, rax.code, rax.size) == ("gp", 0, 8)
    assert (r15.code, r13d.size, r8b.size) == (15, 4, 1)
    assert (xmm3.kind, xmm3.size) == ("xmm", 16)
    assert ah.high and not al.high and ah.code == 4
    assert spl.needs_rex and sil.needs_rex and r12.needs_rex
    assert not ah.needs_rex and not al.needs_rex and not rax.needs_rex
    assert repr(rax) == str(rax) == "rax"


def test_constructors():
    assert gp64(3) is rbx and gp32(1) is ecx and gp8(4) is spl
    assert xmm(3) is xmm3 and ymm(15).name == "ymm15"
    with pytest.raises(EncodeError):
        gp64(16)


@pytest.mark.parametrize(
    "expr, text",
    [
        (qword[rbx + rcx * 8 + 16], "qword ptr [rbx+rcx*8+16]"),
        (qword[8 * rcx + rbx + 16], "qword ptr [rbx+rcx*8+16]"),
        (dword[rbp - 8], "dword ptr [rbp-8]"),
        (byte[rax], "byte ptr [rax]"),
        (word[rax + rbx], "word ptr [rax+rbx*1]"),
        (ptr[rsp + 4], "[rsp+4]"),
        (qword[0x1000], "qword ptr [4096]"),
        (oword[rcx * 4], "xmmword ptr [rcx*4]"),
        (yword[rax + 32], "ymmword ptr [rax+32]"),
        (dword[eax + ecx * 2], "dword ptr [eax+ecx*2]"),
        (qword[16 + rax], "qword ptr [rax+16]"),
        (qword[rax + 8 - 8], "qword ptr [rax]"),
    ],
)
def test_mem_str(expr, text):
    assert str(expr) == repr(expr) == text


def test_mem_fields():
    m = qword[rbx + rcx * 8 + 16]
    assert (m.base, m.index, m.scale, m.disp, m.label, m.size) == (rbx, rcx, 8, 16, None, 8)
    assert ptr[rax].size is None
    assert isinstance(m, MemExpr)


def test_rsp_index_swaps_or_fails():
    m = qword[rax + rsp]
    assert m.base is rsp and m.index is rax
    assert qword[rsp * 1].base is rsp
    with pytest.raises(EncodeError):
        _ = rsp * 2
    with pytest.raises(EncodeError):
        qword[rsp + rsp]


def test_rip_relative_label():
    lbl = Label("tbl")
    m = qword[rip + lbl + 8]
    assert (m.base, m.label, m.disp) == (rip, lbl, 8)
    assert str(m) == "qword ptr [rip+tbl+8]"
    assert str(qword[rip + 16]) == "qword ptr [rip+16]"
    with pytest.raises(EncodeError):
        qword[rax + lbl]
    with pytest.raises(EncodeError):
        qword[lbl]  # type: ignore[index]  # pyright: ignore[reportArgumentType]
    with pytest.raises(EncodeError):
        _ = rip + rcx


def test_mem_validation():
    with pytest.raises(EncodeError):
        _ = rax * 3
    with pytest.raises(EncodeError):
        qword[rax + (1 << 31)]
    qword[rax - (1 << 31)]
    with pytest.raises(EncodeError):
        qword[rax + rbx + rcx]
    with pytest.raises(EncodeError):
        qword[rax + ecx]
    with pytest.raises(EncodeError):
        byte[al]
    with pytest.raises(EncodeError):
        qword[xmm3]
    with pytest.raises(EncodeError, match=r"use qword\[rip \+ label\]"):
        qword["rax"]  # type: ignore[index]  # pyright: ignore[reportArgumentType]  # a str names a label, and labels need a rip base
    with pytest.raises(EncodeError, match="VSIB"):
        qword[rax + xmm3 * 4]
    with pytest.raises(EncodeError, match="VSIB"):
        _ = ymm(1) * 2


def test_absolute_64bit_disp():
    assert qword[0x100000000].disp == 0x100000000
    assert qword[0xFFFFFFFFFFFFFFFF].disp == 0xFFFFFFFFFFFFFFFF
    assert qword[-(1 << 63)].disp == -(1 << 63)
    with pytest.raises(EncodeError):
        qword[1 << 64]
    with pytest.raises(EncodeError, match="int32"):
        _ = qword[0x100000000] + rax
    with pytest.raises(EncodeError, match="int32"):
        qword[rbx + 0x100000000]
    with pytest.raises(EncodeError, match="int32"):
        qword[rip + 0x100000000]


def test_imm():
    assert Imm.coerce(5) == Imm(5) and Imm.coerce(Imm(5)).value == 5
    with pytest.raises(TypeError):
        Imm("5")  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
