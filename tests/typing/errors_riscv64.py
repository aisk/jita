"""riscv64 calls the encoder rejects and the stubs must reject too. Every
line carries the exact ignore comment of both checkers, and unused
ignores are errors, so a line that stops being an error fails the check
(tests/test_typing.py)."""

from jita import Assembler
from jita.riscv64 import *  # noqa: F403

a = Assembler("riscv64")


def calls() -> None:
    add(a0, a1, 1)  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
    add(a0, a1, fa2)  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
    add(a0, a1)  # type: ignore[call-arg]  # pyright: ignore[reportCallIssue]
    addi(a0, a1, a2)  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
    fadd.d(fa0, a1, fa2)  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
    fadd(fa0, fa1, fa2)  # type: ignore[operator]  # pyright: ignore[reportCallIssue]
    fadd.q(fa0, fa1, fa2)  # type: ignore[attr-defined]  # pyright: ignore[reportAttributeAccessIssue]
    fcvt.w.d(fa0, fa0)  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
    fcvt.w.d(a0, fa0, rm="up")  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
    fcvt.d.w(fa0, a0, rm="rtz")  # type: ignore[call-arg]  # pyright: ignore[reportCallIssue]
    ld(a0, a1)  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
    ld(a0, 8)  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
    sd(a0, "lbl")  # type: ignore[call-overload]  # pyright: ignore[reportArgumentType]
    lr.w(a0, a1)  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
    amoadd.w(a0, a1, "lbl")  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
    fence("rw")  # type: ignore[call-overload]  # pyright: ignore[reportCallIssue]
    fence("wr", "rw")  # type: ignore[call-overload]  # pyright: ignore[reportArgumentType]
    csrr(a0, "mstatus")  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
    beq(a0, a1, 8)  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
    j(a0)  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
    call(mem[a0])  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
    li(a0, "x")  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
    ret(a0)  # type: ignore[arg-type, call-arg]  # pyright: ignore[reportCallIssue]
    mv(a0, fa0)  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
    a.add(a0, a1, fa2)  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
    a.fadd.d(fa0, fa1, fa2, asm=a)  # type: ignore[call-arg]  # pyright: ignore[reportCallIssue]
    a.fcvt.w.d(a0, fa0, rm="up")  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
    a.addd(a0, a1, a2)  # type: ignore[operator]  # pyright: ignore[reportCallIssue]


def operands() -> None:
    mem[8]  # type: ignore[index]  # pyright: ignore[reportArgumentType]
    mem[a0, 8]  # type: ignore[index]  # pyright: ignore[reportArgumentType]
    mem[fa0]  # type: ignore[index]  # pyright: ignore[reportArgumentType]
