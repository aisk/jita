"""aarch64 memory operands.

    mem[x0]                     [x0]
    mem[x0 + 8], mem[sp - 16]   [x0, #8], [sp, #-16]
    mem[x0 + x1]                [x0, x1]
    mem[x0 + (x1 << 3)]         [x0, x1, lsl #3]
    mem[x0 + x1.sxtx(3)]        [x0, x1, sxtx #3]
    mem[x0 + w1.uxtw(2)]        [x0, w1, uxtw #2]   (also w1.sxtw(), no amount)
    mem.pre[x0 + 16]            [x0, #16]!          pre-index, writes back
    mem.post[x0, 16]            [x0], #16           post-index, writes back

There are no size prefixes: the access width comes from the instruction
(`ldrb`, `ldrh`, `ldr w0`, `ldr x0`, `ldr d0`, ...). Whether an offset is
encoded scaled (`ldr`) or unscaled (`ldur`) is decided by the encoder from
the value, as DynASM does. Load a label or an Extern's address with
`ldr(x0, lbl)` (pc-relative literal) instead of a memory operand.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..core.errors import EncodeError
from ..core.operand import Operand
from .regs import Mod, Reg, RegMod


def _is_int(x) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


class Addr:
    """An address expression under construction: `x0 + 8`, `x0 + x1 << 3`.

    Not an operand by itself; wrap it with `mem[...]`, `mem.pre[...]`.
    """

    __slots__ = ("base", "index", "mod", "disp")

    def __init__(self, base: Reg, index: Reg | None = None, mod: Mod | None = None, disp: int = 0):
        self.base, self.index, self.mod, self.disp = base, index, mod, disp

    def __add__(self, other) -> Addr:
        if _is_int(other):
            return Addr(self.base, self.index, self.mod, self.disp + other)
        if isinstance(other, (Reg, RegMod)):
            if self.index is not None:
                raise EncodeError(f"too many registers in address: {self} + {other}")
            if isinstance(other, RegMod):
                return Addr(self.base, other.reg, other.mod, self.disp)
            return Addr(self.base, other, None, self.disp)
        return NotImplemented

    def __sub__(self, other) -> Addr:
        if _is_int(other):
            return Addr(self.base, self.index, self.mod, self.disp - other)
        return NotImplemented

    def __lshift__(self, other):
        raise EncodeError("write a scaled index as mem[base + (index << n)]")

    __rshift__ = __lshift__

    def __str__(self) -> str:
        parts = [str(self.base)]
        if self.index is not None:
            parts.append(str(self.index))
        if self.mod is not None:
            parts.append(str(self.mod))
        if self.disp:
            parts.append(str(self.disp))
        return " + ".join(parts)

    __repr__ = __str__


_MODES = ("offset", "pre", "post")


@dataclass(frozen=True, slots=True, repr=False)
class MemExpr(Operand):
    """`[base, #disp]`, `[base, index{, extend}]`, `[base, #disp]!` or
    `[base], #disp`. `mode` is "offset", "pre" or "post". Validated on
    construction; offset ranges depend on the instruction and are checked
    by the encoder."""

    base: Reg
    index: Reg | None = None
    mod: Mod | None = None
    disp: int = 0
    mode: str = "offset"

    def __post_init__(self):
        base, index, mod = self.base, self.index, self.mod
        if not isinstance(base, Reg) or not (base.kind == "sp" or (base.rt == "x" and base.code < 31)):
            raise EncodeError(f"{base} cannot be a memory base (use x0..x30 or sp)")
        if self.mode not in _MODES:
            raise EncodeError(f"bad addressing mode {self.mode!r}")
        if not _is_int(self.disp):
            raise EncodeError(f"offset {self.disp!r} is not an int")
        if index is None:
            if mod is not None:
                raise EncodeError("shift or extend without an index register")
            return
        if self.mode != "offset":
            raise EncodeError(f"{self.mode}-index addressing takes an immediate, not a register")
        if self.disp:
            raise EncodeError("an address cannot have both an index register and an offset")
        if not isinstance(index, Reg) or index.kind != "gp":
            raise EncodeError(f"{index} cannot be a memory index")
        kind = None if mod is None else mod.kind
        if index.rt == "x":
            if kind not in (None, "lsl", "sxtx"):
                raise EncodeError(f"a 64 bit index takes lsl or sxtx, not {kind}")
        elif kind not in ("uxtw", "sxtw"):
            raise EncodeError(f"a 32 bit index needs uxtw or sxtw, e.g. {index}.uxtw()")

    def __str__(self) -> str:
        b = str(self.base)
        if self.mode == "pre":
            return f"[{b}, #{self.disp}]!"
        if self.mode == "post":
            return f"[{b}], #{self.disp}"
        if self.index is not None:
            if self.mod is None:
                return f"[{b}, {self.index}]"
            return f"[{b}, {self.index}, {self.mod}]"
        if self.disp:
            return f"[{b}, #{self.disp}]"
        return f"[{b}]"

    __repr__ = __str__


def _from(x, mode: str) -> MemExpr:
    if isinstance(x, Reg):
        return MemExpr(x, mode=mode)
    if isinstance(x, Addr):
        return MemExpr(x.base, x.index, x.mod, x.disp, mode)
    if isinstance(x, MemExpr) and x.mode == mode:
        return x
    if isinstance(x, (Operand, str)) or _is_int(x):
        raise EncodeError(f"mem[...] expects a base register or base + offset/index, got {x!r}")
    raise TypeError(f"mem[...] expects a register or address expression, got {x!r}")


class _Post:
    __slots__ = ()

    def __getitem__(self, key) -> MemExpr:
        if not (isinstance(key, tuple) and len(key) == 2 and isinstance(key[0], Reg) and _is_int(key[1])):
            raise EncodeError(f"post-index is written mem.post[base, offset], got {key!r}")
        return MemExpr(key[0], disp=key[1], mode="post")

    def __repr__(self) -> str:
        return "mem.post"


class _Pre:
    __slots__ = ()

    def __getitem__(self, key) -> MemExpr:
        return _from(key, "pre")

    def __repr__(self) -> str:
        return "mem.pre"


class _Mem:
    """`mem[...]`, `mem.pre[...]`, `mem.post[base, offset]`."""

    __slots__ = ()
    pre = _Pre()
    post = _Post()

    def __getitem__(self, key) -> MemExpr:
        return _from(key, "offset")

    def __repr__(self) -> str:
        return "mem"


mem = _Mem()

__all__ = ["MemExpr", "mem"]
