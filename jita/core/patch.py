"""Patch kinds and patch records. Linking means applying patches."""

from __future__ import annotations

from dataclasses import dataclass

from .errors import LinkError
from .label import Extern, Label
from .operand import Hole


class PatchKind:
    """How a resolved address is written into a field."""

    __slots__ = ("name", "size")

    def __init__(self, name: str, size: int):
        self.name = name
        self.size = size  # bytes occupied in the section

    def apply(self, buf: bytearray, at: int, target: int, place: int) -> None:
        """Write the field.

        `at` is the offset inside buf, `target` is the absolute address of the
        target (plus addend), `place` is the absolute address of the field
        itself. Raise LinkError on range overflow.
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        return self.name


class AbsKind(PatchKind):
    """Absolute little endian field. Accepts signed or unsigned values."""

    __slots__ = ()

    def apply(self, buf: bytearray, at: int, target: int, place: int) -> None:
        bits = self.size * 8
        if not -(1 << (bits - 1)) <= target < (1 << bits):
            raise LinkError(f"{self.name}: value {target:#x} does not fit")
        buf[at : at + self.size] = (target & ((1 << bits) - 1)).to_bytes(self.size, "little")


class RelKind(PatchKind):
    """Signed displacement relative to the end of the field (x86 style)."""

    __slots__ = ()

    def apply(self, buf: bytearray, at: int, target: int, place: int) -> None:
        bits = self.size * 8
        value = target - (place + self.size)
        if not -(1 << (bits - 1)) <= value < (1 << (bits - 1)):
            raise LinkError(f"{self.name}: displacement {value} out of range")
        buf[at : at + self.size] = value.to_bytes(self.size, "little", signed=True)


ABS8 = AbsKind("abs8", 1)
ABS16 = AbsKind("abs16", 2)
ABS32 = AbsKind("abs32", 4)
ABS64 = AbsKind("abs64", 8)
REL8 = RelKind("rel8", 1)
REL32 = RelKind("rel32", 4)

ABS_BY_SIZE = {1: ABS8, 2: ABS16, 4: ABS32, 8: ABS64}

type PatchTarget = Label | Extern | Hole


@dataclass(slots=True)
class Patch:
    offset: int  # offset within the section
    kind: PatchKind
    target: PatchTarget
    addend: int = 0
