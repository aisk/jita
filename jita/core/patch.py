"""Patch kinds and patch records. Linking means applying patches."""

from __future__ import annotations

from dataclasses import dataclass

from .errors import LinkError
from .labels import Extern, Label
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


class ImmKind(AbsKind):
    """An immediate hole field: like AbsKind, but `target` is the hole's
    value and must lie in `lo..hi`, the range allowed by the instruction
    form the hole was encoded in."""

    __slots__ = ("lo", "hi")

    def __init__(self, name: str, size: int, lo: int, hi: int):
        super().__init__(name, size)
        self.lo, self.hi = lo, hi

    def apply(self, buf: bytearray, at: int, target: int, place: int) -> None:
        if not self.lo <= target <= self.hi:
            raise LinkError(f"value {target} out of range {self.lo}..{self.hi}")
        super().apply(buf, at, target, place)


class BitsKind(PatchKind):
    """Sets a bit field of one byte from a register number (register holes).

    `(value >> take) & mask`, inverted first if `invert`, is written into
    bits `shift .. shift+width-1` of the byte at `at`, replacing what the
    template had there. `forbid` is a value that cannot be encoded in this
    position (rsp as a SIB index). The field is part of an instruction, so
    nothing is reserved for it: `size` is the one byte it modifies.
    """

    __slots__ = ("shift", "width", "take", "invert", "forbid")

    def __init__(self, name: str, shift: int, width: int, take: int = 0, invert: bool = False,
                 forbid: int | None = None):  # fmt: skip
        super().__init__(name, 1)
        self.shift, self.width, self.take = shift, width, take
        self.invert, self.forbid = invert, forbid

    def apply(self, buf: bytearray, at: int, target: int, place: int) -> None:
        if target == self.forbid:
            raise LinkError(f"{self.name}: register number {target} cannot be encoded here")
        mask = (1 << self.width) - 1
        v = target >> self.take
        if self.invert:
            v = ~v
        buf[at] = (buf[at] & ~(mask << self.shift) & 0xFF) | ((v & mask) << self.shift)


_imm_kinds: dict[tuple[int, int, int], ImmKind] = {}
_bits_kinds: dict[tuple, BitsKind] = {}


def imm_kind(size: int, lo: int, hi: int) -> ImmKind:
    """The (shared) ImmKind for a field of `size` bytes accepting lo..hi."""
    kind = _imm_kinds.get((size, lo, hi))
    if kind is None:
        bits = size * 8
        name = f"imm{bits}"
        if lo == -(1 << (bits - 1)) and hi == (1 << (bits - 1)) - 1:
            name += "s"
        elif lo == 0:
            name += "u"
        kind = _imm_kinds[size, lo, hi] = ImmKind(name, size, lo, hi)
    return kind


def bits_kind(name: str, shift: int, width: int, take: int = 0, invert: bool = False,
              forbid: int | None = None) -> BitsKind:  # fmt: skip
    """The (shared) BitsKind with these parameters."""
    key = (name, shift, width, take, invert, forbid)
    kind = _bits_kinds.get(key)
    if kind is None:
        kind = _bits_kinds[key] = BitsKind(*key)
    return kind


ABS8 = AbsKind("abs8", 1)
ABS16 = AbsKind("abs16", 2)
ABS32 = AbsKind("abs32", 4)
ABS64 = AbsKind("abs64", 8)
REL8 = RelKind("rel8", 1)
REL32 = RelKind("rel32", 4)

ABS_BY_SIZE = {1: ABS8, 2: ABS16, 4: ABS32, 8: ABS64}


class SlotKind(PatchKind):
    """A reference to the pointer slot of an Extern, not to the extern.

    Encoders emit it for memory operands such as `qword[rip + ext]`.
    `Assembler.emit_patch` never records it: it replaces the Extern target
    with the assembler's slot label (`Assembler.extern_slot`) and the kind
    with `field`, the kind that actually writes the reference. So
    `call(ext)` stays a direct REL32 to the extern while `call(qword[rip +
    ext])` becomes a REL32 to an 8 byte slot holding its address.
    """

    __slots__ = ("field",)

    def __init__(self, field: PatchKind):
        super().__init__(f"{field.name}@slot", field.size)
        self.field = field

    def apply(self, buf: bytearray, at: int, target: int, place: int) -> None:
        raise LinkError(f"{self.name}: extern slot reference was never resolved to a slot")


SLOT_REL32 = SlotKind(REL32)

type PatchTarget = Label | Extern | Hole


@dataclass(slots=True)
class Patch:
    offset: int  # offset within the section
    kind: PatchKind
    target: PatchTarget
    addend: int = 0
