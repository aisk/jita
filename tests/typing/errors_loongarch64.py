"""loongarch64 calls the encoder rejects and the stubs must reject too.
Every line carries the exact ignore comment of both checkers, and unused
ignores are errors, so a line that stops being an error fails the check
(tests/test_typing.py)."""

from jita import Assembler
from jita.loongarch64 import *  # noqa: F403

a = Assembler("loongarch64")


def calls() -> None:
    add.d(a0, a1, 1)  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
    add.d(a0, a1, fa2)  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
    add.d(a0, a1)  # type: ignore[call-arg]  # pyright: ignore[reportCallIssue]
    addi.d(a0, a1, a2)  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
    fadd.d(fa0, a1, fa2)  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
    add(a0, a1, a2)  # type: ignore[operator]  # pyright: ignore[reportCallIssue]
    add.q(a0, a1, a2)  # type: ignore[attr-defined]  # pyright: ignore[reportAttributeAccessIssue]
    fcmp.clt.d(a0, fa0, fa1)  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
    fcmp.lt.d(fcc0, fa0, fa1)  # type: ignore[attr-defined]  # pyright: ignore[reportAttributeAccessIssue]
    movgr2fcsr(fcc0, a0)  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
    ld.d(a0, 8)  # type: ignore[call-overload]  # pyright: ignore[reportArgumentType]
    ld.d(a0, a1)  # type: ignore[call-overload]  # pyright: ignore[reportArgumentType]
    st.d(a0, "lbl")  # type: ignore[call-overload]  # pyright: ignore[reportCallIssue]
    beq(a0, a1, 8)  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
    b(a0)  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
    call36(a0)  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
    tail36("lbl")  # type: ignore[arg-type, call-arg]  # pyright: ignore[reportCallIssue]
    li.d(a0, "x")  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
    li(a0, 1)  # type: ignore[operator]  # pyright: ignore[reportCallIssue]
    la(a0, "lbl")  # type: ignore[operator]  # pyright: ignore[reportCallIssue]
    ret(a0)  # type: ignore[arg-type, call-arg]  # pyright: ignore[reportCallIssue]
    move(a0, fa0)  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
    fsel(fa0, fa1, fa2, fa3)  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
    a.add.d(a0, a1, fa2)  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
    a.add.d(a0, a1, a2, asm=a)  # type: ignore[call-arg]  # pyright: ignore[reportCallIssue]
    a.addd(a0, a1, a2)  # type: ignore[operator]  # pyright: ignore[reportCallIssue]
    a.revb.h2(a0, a1)  # type: ignore[attr-defined]  # pyright: ignore[reportAttributeAccessIssue]
