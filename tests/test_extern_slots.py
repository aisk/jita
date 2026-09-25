"""Extern pointer slots: `qword[rip + ext]` addresses an 8 byte slot holding
the extern's address, while `call(ext)` stays a direct rel32 call."""

import ctypes
import platform

import pytest
from oracle import assemble, disassemble, requires_oracle

import jita.x64 as x64
from jita import Assembler, EncodeError, Extern, Label, LinkError
from jita.core import ABS64, REL32, SLOT_REL32
from jita.tools.listing import listing
from jita.x64 import *  # noqa: F403

del test  # noqa: F821, the x64 mnemonic, not a pytest test

needs_host = pytest.mark.skipif(
    platform.machine().lower() not in ("x86_64", "amd64") or platform.system() == "Windows",
    reason="needs an x86-64 POSIX host",
)

STRLEN = Extern("strlen")

# (jita instruction, GNU as text with the extern as an undefined symbol)
SLOT_CASES = [
    (lambda e: call(qword[rip + e]), "call qword ptr [rip+{}]"),
    (lambda e: jmp(qword[rip + e]), "jmp qword ptr [rip+{}]"),
    (lambda e: mov(rax, qword[rip + e]), "mov rax, qword ptr [rip+{}]"),
    (lambda e: mov(r11, ptr[rip + e]), "mov r11, qword ptr [rip+{}]"),
    (lambda e: lea(rax, ptr[rip + e]), "lea rax, [rip+{}]"),
    (lambda e: push(qword[rip + e]), "push qword ptr [rip+{}]"),
    (lambda e: cmp(qword[rip + e], 0), "cmp qword ptr [rip+{}], 0"),
]


def build(*fns, ext=STRLEN):
    a = Assembler(x64)
    with a:
        for fn in fns:
            fn(ext)
    return a


def test_slot_created_lazily():
    a = Assembler(x64)
    assert "externs" not in a.sections
    with a:
        call(STRLEN)
    assert "externs" not in a.sections
    assert a.extern_slots == {}
    with a:
        call(qword[rip + STRLEN])
    sec = a.sections["externs"]
    assert sec.align == 8 and not sec.writable
    slot = a.extern_slots["strlen"]
    assert slot.section is sec and slot.offset == 0 and slot.owner is a
    assert slot.name is None
    assert len(sec.buf) == 8
    [p] = sec.patches
    assert (p.offset, p.kind, p.target) == (0, ABS64, STRLEN)


def test_code_patch_targets_the_slot():
    a = build(lambda e: call(qword[rip + e]), lambda e: call(e))
    ind, direct = a.sections["code"].patches
    assert ind.kind is REL32 and ind.target is a.extern_slots["strlen"]
    assert direct.kind is REL32 and direct.target is STRLEN
    assert bytes(a.sections["code"].buf) == bytes.fromhex("ff1500000000e800000000")


def test_one_slot_per_name():
    other = Extern("strlen")
    a = build(*(fn for fn, _ in SLOT_CASES))
    with a:
        call(qword[rip + other])
        mov(rax, qword[rip + Extern("memcpy")])
    assert list(a.extern_slots) == ["strlen", "memcpy"]
    assert len(a.sections["externs"].buf) == 16
    slot = a.extern_slots["strlen"]
    assert a.extern_slot(other) is slot
    code = a.sections["code"].patches
    assert sum(p.target is slot for p in code) == len(SLOT_CASES) + 1


def test_slots_are_per_assembler():
    a, b = build(lambda e: call(qword[rip + e])), build(lambda e: call(qword[rip + e]))
    assert a.extern_slots["strlen"] is not b.extern_slots["strlen"]


def test_extern_slot_method():
    a = Assembler(x64)
    with a:
        nop()
    slot = a.extern_slot(STRLEN)
    assert a.cur is a.sections["code"]
    assert a.extern_slot(STRLEN) is slot
    with pytest.raises(TypeError):
        a.extern_slot("strlen")


def test_slot_alignment_after_manual_data():
    a = Assembler(x64)
    with a.section("externs"):
        a.byte(1)
    slot = a.extern_slot(STRLEN)
    assert slot.offset == 8


def test_slot_kind_requires_an_extern():
    a = Assembler(x64)
    with pytest.raises(TypeError):
        a.emit_patch(SLOT_REL32, Label())


def test_memory_operand_text():
    assert str(qword[rip + STRLEN]) == "qword ptr [rip+strlen]"
    assert str(ptr[rip + STRLEN + 8]) == "[rip+strlen+8]"
    assert qword[rip + STRLEN].label is STRLEN


def test_extern_needs_rip_base():
    with pytest.raises(EncodeError):
        qword[rax + STRLEN]
    with pytest.raises(EncodeError):
        qword[STRLEN]
    with pytest.raises(EncodeError):
        qword[rip + STRLEN + Label()]


def test_link_resolves_slot():
    a = build(lambda e: call(qword[rip + e]), lambda e: mov(rax, qword[rip + e + 8]))
    img = a.link(0x10000, {"strlen": 0x7F0012345678})
    ext = img.section_offsets["externs"]
    assert img.data[ext : ext + 8] == (0x7F0012345678).to_bytes(8, "little")
    slot = img.address(a.extern_slots["strlen"])
    assert slot == 0x10000 + ext
    # call qword [rip+rel]: rip is the end of the 6 byte instruction.
    assert int.from_bytes(img.data[2:6], "little", signed=True) == slot - (0x10000 + 6)
    assert int.from_bytes(img.data[9:13], "little", signed=True) == slot + 8 - (0x10000 + 13)


def test_far_extern_links_through_slot_but_not_direct():
    far = 0x7FFF00000000
    ok = build(lambda e: call(qword[rip + e]))
    ok.link(0x10000, {"strlen": far})
    bad = build(lambda e: call(e))
    with pytest.raises(LinkError):
        bad.link(0x10000, {"strlen": far})


def test_missing_extern_address():
    a = build(lambda e: call(qword[rip + e]))
    with pytest.raises(LinkError, match="extern strlen has no address"):
        a.link(0x10000)


def test_listing_shows_externs_section():
    a = build(lambda e: call(qword[rip + e]))
    with a:
        lbl = Label()
        jmp(lbl)
        label(lbl)
    text = listing(a)
    assert "section externs (align 8, 8 bytes)" in text
    assert "strlen@slot:" in text
    assert "; rel32 -> strlen@slot" in text
    assert "; abs64 -> strlen" in text
    # Slots do not take numbers from other anonymous labels.
    assert "jmp .L1" in text


@requires_oracle
@pytest.mark.parametrize("fn,text", SLOT_CASES)
def test_slot_bytes_match_gnu_as(fn, text):
    a = build(fn)
    assert bytes(a.sections["code"].buf) == assemble(text.format("strlen"))


@requires_oracle
def test_linked_slot_disassembles_to_rip_relative():
    a = build(lambda e: call(qword[rip + e]), lambda e: jmp(qword[rip + e]))
    img = a.link(0x10000, {"strlen": 0x1234})
    code = img.data[: len(a.sections["code"].buf)]
    slot = img.address(a.extern_slots["strlen"]) - 0x10000
    assert disassemble(code) == [
        f"call qword ptr [rip+{slot - 6:#x}]",
        f"jmp qword ptr [rip+{slot - 12:#x}]",
    ]


@needs_host
def test_call_strlen_through_slot():
    libc = ctypes.CDLL(None)
    addr = ctypes.cast(libc.strlen, ctypes.c_void_p).value
    a = Assembler(x64)
    with a:
        sub(rsp, 8)
        call(qword[rip + STRLEN])
        mov(rcx, qword[rip + STRLEN])  # same slot, loaded as a pointer
        add(rsp, 8)
        ret()
    assert len(a.sections["externs"].buf) == 8
    with a.load(externs={"strlen": addr}) as mod:
        fn = mod.function(ctypes.c_size_t, ctypes.c_char_p)
        assert fn(b"hello, slot") == 11
        slot = mod.address(a.extern_slots["strlen"])
        assert ctypes.c_uint64.from_address(slot).value == addr


@needs_host
def test_tail_call_through_slot():
    libc = ctypes.CDLL(None)
    addr = ctypes.cast(libc.strlen, ctypes.c_void_p).value
    a = Assembler(x64)
    with a:
        jmp(qword[rip + Extern("strlen", addr)])
    with a.load() as mod:
        assert mod.function(ctypes.c_size_t, ctypes.c_char_p)(b"abc") == 3
