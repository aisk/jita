"""Patch kinds and patch records. Linking means applying patches."""

from dataclasses import dataclass

from .errors import LinkError
from .labels import Extern, Label


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


class SlotKind(PatchKind):
    """A reference to the pointer slot of an Extern, not to the extern.

    Encoders emit it for memory operands such as `qword[rip + ext]`.
    `Assembler.add_patch` (and so `emit_patch`) never records it: it
    replaces the Extern target with the assembler's slot label
    (`Assembler.extern_slot`) and the kind with `field`, the kind that actually writes the reference. So
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

type PatchTarget = Label | Extern


@dataclass(slots=True)
class Patch:
    offset: int  # offset within the section
    kind: PatchKind
    target: PatchTarget
    addend: int = 0
