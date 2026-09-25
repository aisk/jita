"""aarch64 calls the encoder rejects and the stubs must reject too. Every
line carries the exact ignore comment of both checkers, and unused
ignores are errors, so a line that stops being an error fails the check
(tests/test_typing.py)."""

from jita import Assembler
from jita.aarch64 import *  # noqa: F403
from jita.aarch64 import Mod

a = Assembler("aarch64")


def calls() -> None:
    add(x0, w1, x2)  # type: ignore[call-overload]  # pyright: ignore[reportArgumentType, reportCallIssue]
    add(w0, w1, x2)  # type: ignore[call-overload]  # pyright: ignore[reportArgumentType, reportCallIssue]
    add(x0, x1, s2)  # type: ignore[call-overload]  # pyright: ignore[reportArgumentType, reportCallIssue]
    add(x0, x1)  # type: ignore[call-overload]  # pyright: ignore[reportCallIssue]
    sub(w0, w1, x2 << 3)  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType, reportCallIssue]
    mov(x0, d1)  # type: ignore[call-overload]  # pyright: ignore[reportArgumentType, reportCallIssue]
    fadd(d0, s1, d2)  # type: ignore[call-overload]  # pyright: ignore[reportArgumentType, reportCallIssue]
    fadd(d0, d1, x2)  # type: ignore[call-overload]  # pyright: ignore[reportArgumentType, reportCallIssue]
    fmov(x0, 1.5)  # type: ignore[call-overload]  # pyright: ignore[reportArgumentType, reportCallIssue]
    ldr(x0, x1)  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
    ldr(x0, 8)  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
    str_(x0, "lbl")  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
    ldp(x0, mem[x1])  # type: ignore[call-overload]  # pyright: ignore[reportCallIssue]
    stp(x0, w1, mem[sp])  # type: ignore[call-overload]  # pyright: ignore[reportArgumentType, reportCallIssue]
    csel(x0, x1, x2, "lx")  # type: ignore[call-overload]  # pyright: ignore[reportArgumentType, reportCallIssue]
    csel(x0, x1, x2, 3)  # type: ignore[call-overload]  # pyright: ignore[reportArgumentType, reportCallIssue]
    cset(w0, "ne", "eq")  # type: ignore[arg-type, call-arg]  # pyright: ignore[reportCallIssue]
    b(x0)  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
    b.eq(x0)  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
    bl(mem[x0])  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
    br("lbl")  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
    cbz(d0, "lbl")  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
    tbz(x0, "lbl", 3)  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
    movz(x0, 1, lsl="16")  # type: ignore[call-overload]  # pyright: ignore[reportArgumentType]
    movz(x0, x1)  # type: ignore[call-overload]  # pyright: ignore[reportArgumentType]
    bti("x")  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
    ret(1)  # type: ignore[call-overload]  # pyright: ignore[reportArgumentType]
    lsl(x0, x1, "3")  # type: ignore[call-overload]  # pyright: ignore[reportArgumentType, reportCallIssue]
    a.add(x0, w1, x2)  # type: ignore[call-overload]  # pyright: ignore[reportArgumentType, reportCallIssue]
    a.b.eq(x0)  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
    a.str(x0, x1)  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
    a.movz(x0, 1, lsl=16, asm=a)  # type: ignore[call-overload]  # pyright: ignore[reportCallIssue]


def operands() -> None:
    mem[8]  # type: ignore[index]  # pyright: ignore[reportArgumentType]
    mem[x0, 8]  # type: ignore[index]  # pyright: ignore[reportArgumentType]
    mem.post[x0 + 8]  # type: ignore[index]  # pyright: ignore[reportArgumentType]
    mem.pre[x0, 8]  # type: ignore[index]  # pyright: ignore[reportArgumentType]
    _ = x0 << x1  # type: ignore[operator]  # pyright: ignore[reportOperatorIssue]
    _ = x0.lsl("3")  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
    _ = (x0 + 8) + mem[x1]  # type: ignore[operator]  # pyright: ignore[reportOperatorIssue]
    gp64("3")  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
    Mod("lsl", "3")  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]


def names() -> None:
    addd(x0, x1, x2)  # type: ignore[name-defined]  # pyright: ignore[reportUndefinedVariable]
    b.eqq("lbl")  # type: ignore[attr-defined]  # pyright: ignore[reportAttributeAccessIssue]
    a.b.eqq("lbl")  # type: ignore[attr-defined]  # pyright: ignore[reportAttributeAccessIssue]
