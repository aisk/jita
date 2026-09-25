"""aarch64 fragments: label and extern references replayed on instantiation."""

from oracle import aarch64_disassemble, requires_aarch64_disassembler

import jita.aarch64 as A
from jita import Assembler, Extern, Fragment, Label, label
from jita.aarch64 import *  # noqa: F403


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
