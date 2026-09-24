"""Architecture neutral operand base classes."""

from __future__ import annotations


class Operand:
    """Base class for all operands. Arch packages subclass it.

    `__str__` must produce the arch's native assembler syntax.
    """

    __slots__ = ()


class Register(Operand):
    """A machine register. Identity is the register object itself."""

    __slots__ = ("name", "kind", "code", "size")

    def __init__(self, name: str, kind: str, code: int, size: int):
        self.name = name
        self.kind = kind  # arch-defined class, e.g. "gp", "xmm", "ymm"
        self.code = code  # hardware encoding number
        self.size = size  # width in bytes

    def __repr__(self) -> str:
        return self.name

    def __str__(self) -> str:
        return self.name


class Imm(Operand):
    """An immediate value. Plain ints are accepted wherever an Imm is."""

    __slots__ = ("value",)

    def __init__(self, value: int):
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(f"immediate must be an int, got {value!r}")
        self.value = value

    @classmethod
    def coerce(cls, x: Imm | int) -> Imm:
        return x if isinstance(x, Imm) else cls(x)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Imm):
            return self.value == other.value
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self.value)

    def __repr__(self) -> str:
        return f"Imm({self.value})"

    def __str__(self) -> str:
        return str(self.value)


class Hole(Operand):
    """Placeholder for phase-2 templates. Only the class exists in v1."""

    __slots__ = ("name",)

    def __init__(self, name: str | None = None):
        self.name = name

    def __repr__(self) -> str:
        return f"Hole({self.name!r})"
