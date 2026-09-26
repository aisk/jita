"""x64 calls the encoder rejects and the stubs must reject too. Every
line carries the exact ignore comment of both checkers, and unused
ignores are errors, so a line that stops being an error fails the check
(tests/test_typing.py)."""

from jita import Assembler, Label
from jita.x64 import *  # noqa: F403
from jita.x64 import MemExpr

a = Assembler("x64")


def calls() -> None:
    add(rax, ecx)  # type: ignore[call-overload]  # pyright: ignore[reportArgumentType, reportCallIssue]
    mov(eax, qword[rdi])  # type: ignore[call-overload]  # pyright: ignore[reportArgumentType, reportCallIssue]
    mov(qword[rdi], eax)  # type: ignore[call-overload]  # pyright: ignore[reportArgumentType, reportCallIssue]
    movzx(eax, ecx)  # type: ignore[call-overload]  # pyright: ignore[reportArgumentType, reportCallIssue]
    movzx(eax, ptr[rdi])  # type: ignore[call-overload]  # pyright: ignore[reportArgumentType, reportCallIssue]
    movsx(rax, qword[rdi])  # type: ignore[call-overload]  # pyright: ignore[reportArgumentType, reportCallIssue]
    add(ptr[rdi], 1)  # type: ignore[call-overload]  # pyright: ignore[reportArgumentType, reportCallIssue]
    push(ptr[rax])  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
    push(eax)  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
    jmp(eax)  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
    jmp(ptr[rax])  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
    jmp(1)  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
    call(ecx)  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
    lea(rax, rdi)  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
    lea(al, ptr[rdi])  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
    shl(rax, ecx)  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
    mov(eax, "table")  # type: ignore[call-overload]  # pyright: ignore[reportArgumentType, reportCallIssue]
    mov(eax, Label())  # type: ignore[call-overload]  # pyright: ignore[reportArgumentType, reportCallIssue]
    imul(eax, rdi, 3)  # type: ignore[call-overload]  # pyright: ignore[reportArgumentType, reportCallIssue]
    imul(al, bl)  # type: ignore[call-overload]  # pyright: ignore[reportArgumentType, reportCallIssue]
    addsd(xmm0, dword[rdi])  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
    addsd(xmm0, rax)  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
    movq(xmm0, eax)  # type: ignore[call-overload]  # pyright: ignore[reportArgumentType, reportCallIssue]
    vaddps(ymm0, ymm1, xmm2)  # type: ignore[call-overload]  # pyright: ignore[reportArgumentType, reportCallIssue]
    fadd(xmm0)  # type: ignore[call-overload]  # pyright: ignore[reportArgumentType]
    mov(rax)  # type: ignore[call-overload]  # pyright: ignore[reportCallIssue]
    ret(1, 2)  # type: ignore[call-overload]  # pyright: ignore[reportCallIssue]
    nop(rax)  # type: ignore[arg-type, call-arg]  # pyright: ignore[reportCallIssue]
    jz(rax)  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
    jz.short(rax)  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
    jmp.short(qword[rax])  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
    mov(rip, 1)  # type: ignore[call-overload]  # pyright: ignore[reportArgumentType, reportCallIssue]
    mov(rax, MemExpr(base=rdi, size=8))  # type: ignore[call-overload]  # pyright: ignore[reportArgumentType, reportCallIssue]
    a.mov(eax, rbx)  # type: ignore[call-overload]  # pyright: ignore[reportArgumentType, reportCallIssue]
    a.mov(rax, 1, asm=a)  # type: ignore[call-overload]  # pyright: ignore[reportCallIssue]
    a.jmp.short(rax)  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
    mov(rax, 1, asm=a, lock=True)  # type: ignore[call-overload]  # pyright: ignore[reportCallIssue]


def operands() -> None:
    qword["tbl"]  # type: ignore[index]  # pyright: ignore[reportArgumentType]
    qword[Label()]  # type: ignore[index]  # pyright: ignore[reportArgumentType]
    _ = rax * rbx  # type: ignore[operator]  # pyright: ignore[reportOperatorIssue]
    _ = rax - rbx  # type: ignore[operator]  # pyright: ignore[reportOperatorIssue]
    gp64("3")  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]


def names() -> None:
    movv(rax, 1)  # type: ignore[name-defined]  # pyright: ignore[reportUndefinedVariable]
    a.movv(rax, 1)  # type: ignore[operator]  # pyright: ignore[reportCallIssue]
    a.jz.shortt("x")  # type: ignore[attr-defined]  # pyright: ignore[reportAttributeAccessIssue]
