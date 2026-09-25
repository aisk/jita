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


# Accepted value range of an immediate hole, by size in bytes. Like
# immediates elsewhere, 8/16/32 bit holes accept signed or unsigned values;
# each use in an instruction may narrow the range further.
_IMM_RANGE = {
    1: (-(1 << 7), (1 << 8) - 1),
    2: (-(1 << 15), (1 << 16) - 1),
    4: (-(1 << 31), (1 << 32) - 1),
    8: (-(1 << 63), (1 << 64) - 1),
}


class Hole(Operand):
    """A placeholder operand in a Fragment, filled in when it is instantiated.

    Holes are created with typed constructors, which fix everything the
    encoder needs to choose an instruction form:

        Hole.gp8(name) .. Hole.gp64(name), Hole.xmm(name), Hole.ymm(name)
        Hole.fp32(name), Hole.fp64(name)          (aarch64)
        Hole.imm8(name) .. Hole.imm64(name)
        Hole.label(name)

    `kind` is "reg", "imm" or "label". Register holes have a `regclass`
    ("gp", "xmm", "ymm", "fp") and a `size` in bytes; immediate holes a
    `size`.
    `name` is the keyword that supplies the value to
    `Fragment.instantiate`. Holes are hashable by identity.

    A gp64 register hole also builds memory operands like a register does:
    `qword[src + 8]`, `qword[rax + idx*8]` on x64, `mem[src + 8]` on
    aarch64.
    """

    __slots__ = ("name", "kind", "regclass", "size")

    def __init__(self, name: str | None = None):
        raise TypeError(
            "Hole needs a typed constructor: Hole.gp64(name), Hole.xmm(name), "
            "Hole.imm32(name), Hole.label(name), ..."
        )

    @classmethod
    def _new(cls, name: str, kind: str, regclass: str | None, size: int | None) -> Hole:
        if not isinstance(name, str) or not name.isidentifier():
            raise TypeError(f"hole name must be a Python identifier, got {name!r}")
        h = object.__new__(cls)
        h.name, h.kind, h.regclass, h.size = name, kind, regclass, size
        return h

    @classmethod
    def gp8(cls, name: str) -> Hole:
        """An 8 bit general purpose register (never ah/ch/dh/bh)."""
        return cls._new(name, "reg", "gp", 1)

    @classmethod
    def gp16(cls, name: str) -> Hole:
        return cls._new(name, "reg", "gp", 2)

    @classmethod
    def gp32(cls, name: str) -> Hole:
        return cls._new(name, "reg", "gp", 4)

    @classmethod
    def gp64(cls, name: str) -> Hole:
        return cls._new(name, "reg", "gp", 8)

    @classmethod
    def xmm(cls, name: str) -> Hole:
        return cls._new(name, "reg", "xmm", 16)

    @classmethod
    def ymm(cls, name: str) -> Hole:
        return cls._new(name, "reg", "ymm", 32)

    @classmethod
    def fp32(cls, name: str) -> Hole:
        """A single precision FP register (aarch64 s0..s31)."""
        return cls._new(name, "reg", "fp", 4)

    @classmethod
    def fp64(cls, name: str) -> Hole:
        """A double precision FP register (aarch64 d0..d31)."""
        return cls._new(name, "reg", "fp", 8)

    @classmethod
    def imm8(cls, name: str) -> Hole:
        return cls._new(name, "imm", None, 1)

    @classmethod
    def imm16(cls, name: str) -> Hole:
        return cls._new(name, "imm", None, 2)

    @classmethod
    def imm32(cls, name: str) -> Hole:
        return cls._new(name, "imm", None, 4)

    @classmethod
    def imm64(cls, name: str) -> Hole:
        return cls._new(name, "imm", None, 8)

    @classmethod
    def label(cls, name: str) -> Hole:
        """A branch target, rip-relative address or data address."""
        return cls._new(name, "label", None, None)

    @property
    def range(self) -> tuple[int, int]:
        """Inclusive value range of an immediate hole."""
        if self.kind != "imm":
            raise TypeError(f"{self!r} is not an immediate hole")
        return _IMM_RANGE[self.size]

    def __repr__(self) -> str:
        if self.kind == "reg":
            ctor = f"{self.regclass}{self.size * 8}" if self.regclass in ("gp", "fp") else self.regclass
        elif self.kind == "imm":
            ctor = f"imm{self.size * 8}"
        else:
            ctor = "label"
        return f"Hole.{ctor}({self.name!r})"

    def __str__(self) -> str:
        return self.name

    # Register holes build memory operands through the architecture of the
    # current assembler (the host architecture outside of any context).

    def _mem(self, scale: int | None = None):
        if self.kind != "reg":
            return NotImplemented
        from .assembler import _current, _host_arch

        asm = _current.get()
        arch = asm.arch if asm is not None else _host_arch()
        build = getattr(arch, "hole_mem", None)
        if build is None:
            raise TypeError(f"{arch.name} has no memory operands with register holes")
        return build(self, scale)

    def __add__(self, other):
        m = self._mem()
        return m if m is NotImplemented else m + other

    __radd__ = __add__

    def __sub__(self, other):
        m = self._mem()
        return m if m is NotImplemented else m - other

    def __mul__(self, scale):
        if not isinstance(scale, int) or isinstance(scale, bool):
            return NotImplemented
        return self._mem(scale)

    __rmul__ = __mul__
