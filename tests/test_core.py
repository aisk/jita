import pytest

import jita.x64 as x64
from jita import Assembler, Extern, JitaError, Label, LinkError, current, label
from jita.core import ABS32, REL8, REL32, EncodeError
from jita.core.patch import SLOT_REL32
from jita.tools.listing import listing


def jmp32(a, lbl):
    a.bytes(b"\xe9")
    a.emit_patch(REL32, lbl)


def test_default_section_and_arch():
    a = Assembler(x64)
    assert a.arch is x64.ARCH
    assert list(a.sections) == ["code"] and a.cur.name == "code"
    assert Assembler(x64.ARCH).arch is x64.ARCH
    assert Assembler().arch is x64.ARCH


def test_data_directives():
    a = Assembler(x64)
    a.byte(1, 0xFF, -1)
    a.word(0x1234)
    a.dword(-2)
    a.qword(0x1122334455667788)
    a.space(3, 0xAA)
    assert a.cur.buf == bytes.fromhex("01ffff 3412 feffffff 8877665544332211 aaaaaa".replace(" ", ""))
    with pytest.raises(EncodeError):
        a.byte(256)
    with pytest.raises(EncodeError):
        a.word(-40000)
    with pytest.raises(ValueError):
        a.space(-1)


def test_align_nop_and_fill():
    a = Assembler(x64)
    x64.ret(asm=a)
    a.align(8)
    assert a.cur.buf == b"\xc3" + x64.ARCH.nop_fill(7)
    a.bytes(b"\x00")
    a.align(4, b"\xcc")
    assert a.cur.buf[9:] == b"\xcc" * 3
    a.align(64)
    assert a.cur.align == 64 and a.pos() == 64
    with pytest.raises(ValueError):
        a.align(3)


def test_nop_fill_lengths():
    for n in range(40):
        assert len(x64.ARCH.nop_fill(n)) == n


def test_rel32_forward_and_backward():
    a = Assembler(x64)
    fwd = Label()
    jmp32(a, fwd)  # 0: e9 -> fwd
    back = a.label()  # 5
    a.bytes(b"\x90")
    a.bind(fwd)  # 6
    jmp32(a, back)  # 6: e9 -> 5
    img = a.link()
    assert img.data == bytes.fromhex("e901000000" "90" "e9faffffff")
    # Linking at a different base does not change pc-relative code.
    assert a.link(base=0x7F0000000000).data == img.data


def test_label_here_uses_current():
    a = Assembler(x64)
    with a:
        a.bytes(b"\x90\x90")
        lbl = label(Label("x"))
    assert lbl.section is a.cur and lbl.offset == 2
    assert a.link(0x1000).address("x") == 0x1002


def test_multiple_sections_alignment_and_abs64():
    a = Assembler(x64)
    a.bytes(b"\x90" * 3)
    with a.section("data", align=32) as sec:
        assert a.cur is sec
        a.bytes(b"\x01")
        tbl = a.label("table")
        a.qword(0x55)
        a.qword(tbl)
    assert a.cur.name == "code"
    a.bytes(b"\xc3")
    img = a.link(base=0x10000)
    assert img.section_offsets == {"code": 0, "data": 32}
    assert img.symbols == {"table": 33}
    assert img.address(tbl) == 0x10000 + 33
    assert img.data[:4] == b"\x90\x90\x90\xc3"
    assert img.data[4:32] == bytes(28)
    assert img.data[41:49] == (0x10000 + 33).to_bytes(8, "little")


def test_section_plain_call_switches_permanently():
    a = Assembler(x64)
    a.section("rodata")
    assert a.cur.name == "rodata"
    a.section("code")
    assert list(a.sections) == ["code", "rodata"]


def test_extern_resolution():
    a = Assembler(x64)
    a.qword(Extern("f"), Extern("g", 0x2000))
    a.dword(Extern("g", 0x2000))
    img = a.link(externs={"f": 0x1234})
    assert img.data == (0x1234).to_bytes(8, "little") + (0x2000).to_bytes(8, "little") + (0x2000).to_bytes(4, "little")
    # The mapping wins over a fixed address.
    assert a.link(externs={"f": 1, "g": 2}).data[8:16] == (2).to_bytes(8, "little")
    with pytest.raises(LinkError, match="extern f"):
        a.link()


def test_unbound_label():
    a = Assembler(x64)
    jmp32(a, Label("nowhere"))
    with pytest.raises(LinkError, match="nowhere is never bound"):
        a.link()


def test_double_bind():
    a = Assembler(x64)
    lbl = a.label()
    with pytest.raises(LinkError):
        a.bind(lbl)
    a.label("dup")
    with pytest.raises(LinkError):
        a.label("dup")


def test_rel8_range():
    a = Assembler(x64)
    top = a.label()
    a.bytes(b"\x90" * 126)
    a.bytes(b"\xeb")
    a.emit_patch(REL8, top)  # -128: fits
    assert a.link().data[-1] == 0x80
    a.bytes(b"\xeb")
    a.emit_patch(REL8, top)  # -130: overflow
    with pytest.raises(LinkError, match=r"code\+0x81"):
        a.link()


def test_abs32_overflow_and_addend():
    a = Assembler(x64)
    lbl = a.label()
    a.emit_patch(ABS32, lbl, addend=4)
    assert a.link(base=0x100).data == (0x104).to_bytes(4, "little")
    with pytest.raises(LinkError):
        a.link(base=1 << 40)


def test_patch_target_must_be_a_label_or_extern():
    a = Assembler(x64)
    with pytest.raises(TypeError):
        a.emit_patch(ABS32, 5)
    assert a.cur.buf == b"" and a.cur.patches == []


def test_add_patch_is_the_primitive():
    # emit_patch reserves the bytes and rolls them back when add_patch
    # rejects the patch; add_patch records over bytes already emitted.
    a = Assembler(x64)
    with pytest.raises(TypeError):
        a.emit_patch(ABS32, 5)
    with pytest.raises(TypeError):
        a.emit_patch(SLOT_REL32, Label())
    assert a.cur.buf == b"" and a.cur.patches == []
    a.emit(b"\x90" * 8)
    with pytest.raises(ValueError, match="outside"):
        a.add_patch(6, ABS32, Label())
    lbl = a.label("t")
    a.add_patch(4, ABS32, lbl)
    a.add_patch(0, SLOT_REL32, Extern("e"))
    assert a.cur.patches[0].target is lbl
    p = a.cur.patches[1]
    assert p.kind is REL32 and p.target is a.extern_slots["e"]
    assert a.cur.buf == b"\x90" * 8  # nothing reserved


def test_label_from_other_assembler():
    a, b = Assembler(x64), Assembler(x64)
    lbl = b.label()
    jmp32(a, lbl)
    with pytest.raises(LinkError, match="another assembler"):
        a.link()


def test_pc_labels():
    a = Assembler(x64)
    assert len(a.pc) == 0
    assert a.pc[3] is a.pc[3]
    assert len(a.pc) == 4
    jmp32(a, a.pc[3])
    a.bind(a.pc[3])
    assert a.link().data == bytes.fromhex("e900000000")
    assert "pc[3]" in repr(a.pc[3])
    with pytest.raises(IndexError):
        a.pc[-1]


def test_context_nesting():
    with pytest.raises(JitaError, match="no active Assembler"):
        current()
    outer, inner = Assembler(x64), Assembler(x64)
    with outer:
        assert current() is outer
        with inner:
            assert current() is inner
            with outer:
                assert current() is outer
            assert current() is inner
        assert current() is outer
    with pytest.raises(JitaError):
        current()


def test_labels_hash_by_identity():
    assert Label("a") != Label("a")
    assert len({Label("a"), Label("a")}) == 2


def test_missing_mnemonic():
    a = Assembler(x64)
    with pytest.raises(AttributeError):
        a.definitely_not_an_insn


def test_method_delegation_binds_asm():
    calls = []

    def fake(*ops, asm=None):
        calls.append((ops, asm))

    fake.short = lambda *ops, asm=None: calls.append(("short", ops, asm))

    class FakeArch(x64.X64Arch):
        insns = {"fake": fake}

    a = Assembler(FakeArch())
    a.fake(1, 2)
    a.fake.short(3)
    assert calls == [((1, 2), a), ("short", (3,), a)]


def test_listing():
    a = Assembler(x64)
    a.label("entry")
    a.bytes(b"\xb8\x2a\x00\x00\x00")
    tgt = Label("tgt")
    jmp32(a, tgt)
    with a.section("data"):
        a.bind(tgt)
        a.qword(tgt)
    text = listing(a)
    assert "section code" in text and "section data" in text
    assert "entry:" in text and "tgt:" in text
    assert "rel32 -> tgt" in text and "abs64 -> tgt" in text
    assert "b8 2a 00 00 00" in text
    linked = listing(a, a.link(base=0x4000))
    assert "00004000" in linked


def test_listing_one_insn_per_line():
    a = Assembler(x64)
    fwd, k = Label(), Label()
    a.label("entry")
    a.mov(x64.rax, x64.qword[x64.rip + k])
    a.jmp(fwd)
    a.jmp.short(fwd)
    a.label()  # anonymous, numbered .L1 in bind order
    a.bind(fwd)
    a.ret()
    with a.section("data"):
        a.bind(k)
        a.qword(fwd)
    lines = listing(a).splitlines()
    assert lines[1:] == [
        "entry:",
        "  00000000  48 8b 05 00 00 00 00     mov rax, qword ptr [rip+.L3]    ; rel32 -> .L3",
        "  00000007  e9 00 00 00 00           jmp .L2                         ; rel32 -> .L2",
        "  0000000c  eb 00                    jmp.short .L2                   ; rel8 -> .L2",
        ".L1:",
        ".L2:",
        "  0000000e  c3                       ret",
        "section data (align 16, 8 bytes)",
        ".L3:",
        "  00000000  00 00 00 00 00 00 00 00  ; abs64 -> .L2",
    ]


def test_listing_long_insn_continues():
    a = Assembler(x64)
    a.mov64(x64.rax, 0x1122334455667788)
    lines = listing(a).splitlines()
    assert lines[1] == "  00000000  48 b8 88 77 66 55 44 33  mov64 rax, 0x1122334455667788"
    assert lines[2] == "  00000008  22 11"


def test_concurrent_contexts_in_asyncio_tasks():
    import asyncio

    a, b = Assembler(x64), Assembler(x64)

    async def worker(asm, n, log):
        with asm:
            await asyncio.sleep(0)
            assert current() is asm
            current().bytes(bytes([n]))
            await asyncio.sleep(0)
            log.append(n)
        # Exit in the other task must not have touched our context.
        with pytest.raises(JitaError):
            current()

    async def main():
        log = []
        # Two tasks on the same assembler, interleaving enter/exit, plus one
        # on a different assembler.
        await asyncio.gather(worker(a, 1, log), worker(a, 2, log), worker(b, 3, log))
        return log

    assert sorted(asyncio.run(main())) == [1, 2, 3]
    assert sorted(bytes(a.cur.buf)) == [1, 2] and bytes(b.cur.buf) == b"\x03"
    with pytest.raises(JitaError):
        current()


def test_nested_contexts():
    a, b = Assembler(x64), Assembler(x64)
    with a:
        with b:
            assert current() is b
            with a:
                assert current() is a
            assert current() is b
        assert current() is a
    with pytest.raises(JitaError):
        a.__exit__(None, None, None)


def test_pc_label_binds_into_owner():
    a, b = Assembler(x64), Assembler(x64)
    a.bytes(b"\x90")
    lbl = a.label(a.pc[0])
    assert lbl is a.pc[0] and lbl.section is a.cur and lbl.offset == 1
    # Owned labels refuse to be defined in another assembler.
    with b:
        b.bytes(b"\x90" * 4)
        with pytest.raises(LinkError, match="another assembler"):
            label(a.pc[1])
    assert not a.pc[1].bound
    assert not b.labels
    # Owned labels cannot be bound into another assembler.
    with pytest.raises(LinkError, match="another assembler"):
        b.bind(a.pc[2])
    assert a.label().owner is a


def test_label_directive_uses_current():
    a = Assembler(x64)
    lbl = Label()
    assert lbl.owner is None
    with pytest.raises(JitaError, match="no active Assembler"):
        label(lbl)
    with a:
        a.bytes(b"\x90")
        assert label(lbl) is lbl
        anon = label()
        named = label("n")
    assert lbl.section is a.cur and lbl.offset == 1
    assert anon.owner is a and anon.name is None and anon.offset == 1
    assert named.name == "n" and a.named("n") is named
    with pytest.raises(TypeError):
        a.label(3)  # type: ignore[arg-type]
    assert not hasattr(lbl, "here")


def test_align_zero_fills_data_sections():
    a = Assembler(x64)
    a.bytes(b"\x01")
    a.align(8)
    assert a.cur.buf == b"\x01" + bytes(7)
    x64.ret(asm=a)
    a.align(4)
    assert a.cur.buf[8:] == b"\xc3" + x64.ARCH.nop_fill(3)
    with a.section("data"):
        a.byte(1)
        a.align(4)
        a.align(8, b"\xcc")
    assert a.sections["data"].buf == b"\x01" + bytes(3) + b"\xcc" * 4


def test_image_address_rejects_label_bound_after_link():
    a = Assembler(x64)
    a.bytes(b"\x90" * 4)
    end = a.label("end")  # bound before linking at the section end
    img = a.link(0x1000)
    assert img.section_sizes == {"code": 4}
    assert img.address(end) == 0x1004
    late_end = a.label()  # same offset, but bound after linking
    with pytest.raises(LinkError, match="bound after linking"):
        img.address(late_end)
    a.bytes(b"\x90")
    with pytest.raises(LinkError, match="bound after linking"):
        img.address(a.label())
    with a.section("data"):
        with pytest.raises(LinkError, match="bound after linking"):
            img.address(a.label())


def test_star_exports():
    ns = {}
    exec("from jita import *", ns)
    names = set(ns) - {"__builtins__"}
    assert {"Assembler", "Label", "Extern", "Image", "Module", "PcLabels", "current"} <= names
    assert {"LinkError", "LoadError", "EncodeError", "JitaError"} <= names
    ns = {}
    exec("from jita.core import *", ns)
    names = set(ns) - {"__builtins__"}
    assert {"Assembler", "Section", "Image", "link", "ABS64", "REL32"} <= names
    # No submodules leak through the star import.
    import types

    assert not [n for n in names if isinstance(ns[n], types.ModuleType)]
    assert not names & {"arch", "assembler", "errors", "labels", "operand", "patch", "section"}
    assert callable(ns["label"]) and isinstance(ns["Label"], type)


def test_named_label_forward_reference():
    a = Assembler(x64)
    fwd = a.named("done")
    assert not fwd.bound and fwd.owner is a
    assert a.named("done") is fwd  # same name, same label
    jmp32(a, fwd)
    a.bytes(b"\x90")
    assert a.label("done") is fwd  # binds the forward reference
    img = a.link(0x1000)
    assert img.address("done") == 0x1006
    assert img.data[:5] == bytes.fromhex("e901000000")
    assert a.named("done") is fwd  # bound labels are found by name too


def test_named_label_in_data_directives():
    a = Assembler(x64)
    a.qword("tbl")
    a.dword("tbl", 7)
    a.label("tbl")
    img = a.link(0x1000)
    assert img.data[:8] == (0x1010).to_bytes(8, "little")
    assert img.data[8:12] == (0x1010).to_bytes(4, "little")


def test_named_label_bound_through_named():
    a = Assembler(x64)
    a.qword("x")
    with a:
        a.bytes(b"\x90")
        a.label(a.named("x"))
    assert a.link(0).address("x") == 9


def test_named_label_never_bound():
    a = Assembler(x64)
    a.qword("nowhere")
    with pytest.raises(LinkError, match="nowhere is never bound"):
        a.link()


def test_named_label_rejects_binding_a_different_object():
    a = Assembler(x64)
    a.qword("x")
    with pytest.raises(LinkError, match="already referenced by name"):
        a.bind(Label("x"))
    a.label("x")  # the forward reference itself still binds fine
    with pytest.raises(LinkError, match="already bound"):
        a.label("x")
    with pytest.raises(LinkError, match="duplicate label name"):
        a.bind(Label("x"))


def test_named_label_rejects_bad_names():
    a = Assembler(x64)
    with pytest.raises(TypeError):
        a.named("")
    with pytest.raises(TypeError):
        a.named(3)  # type: ignore[arg-type]
