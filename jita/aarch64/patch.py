"""aarch64 relocation kinds.

Every kind covers one 32 bit instruction word. The encoder emits the word
with the field left zero and the kind ORs the resolved field into it at
link time. `place` is the address of the instruction itself; displacements
are `target - place` (no end-of-field correction as on x86). Out of range
or misaligned displacements raise LinkError.

    REL26       b, bl                       imm26 << 2, +-128MB
    REL19       b.cond, cbz, cbnz, ldr lit  imm19 << 2 at bit 5, +-1MB
    REL14       tbz, tbnz                   imm14 << 2 at bit 5, +-32KB
    REL21_ADR   adr                         byte offset, lo2 at 29, hi19 at 5
    REL21_ADRP  adrp                        4KB page delta, same layout

Absolute data (`a.qword(label)`) uses the core ABS kinds. Register holes
of a Fragment use `RegField`, a core `BitsKind` for one byte of a 5 bit
register field.
"""

from __future__ import annotations

from ..core.errors import LinkError
from ..core.patch import BitsKind, PatchKind


def _or_word(buf: bytearray, at: int, bits: int) -> None:
    word = int.from_bytes(buf[at : at + 4], "little") | bits
    buf[at : at + 4] = word.to_bytes(4, "little")


class BranchKind(PatchKind):
    """Word aligned pc-relative field of `bits` bits at bit `pos`."""

    __slots__ = ("bits", "pos")

    def __init__(self, name: str, bits: int, pos: int):
        super().__init__(name, 4)
        self.bits, self.pos = bits, pos

    def apply(self, buf: bytearray, at: int, target: int, place: int) -> None:
        n = target - place
        if n & 3:
            raise LinkError(f"{self.name}: displacement {n} is not a multiple of 4")
        limit = 1 << (self.bits + 1)
        if not -limit <= n < limit:
            raise LinkError(f"{self.name}: displacement {n} out of range")
        _or_word(buf, at, ((n >> 2) & ((1 << self.bits) - 1)) << self.pos)


class AdrKind(PatchKind):
    """adr/adrp: 21 bit signed value split into immlo (29-30) and immhi
    (5-23). For adrp the value is the 4KB page delta."""

    __slots__ = ("page",)

    def __init__(self, name: str, page: bool):
        super().__init__(name, 4)
        self.page = page

    def apply(self, buf: bytearray, at: int, target: int, place: int) -> None:
        n = (target >> 12) - (place >> 12) if self.page else target - place
        if not -(1 << 20) <= n < (1 << 20):
            raise LinkError(f"{self.name}: displacement {n} out of range")
        _or_word(buf, at, ((n & 3) << 29) | (((n >> 2) & 0x7FFFF) << 5))


class RegField(BitsKind):
    """One byte's part of a 5 bit register field, filled from a register
    hole. General purpose fields forbid 31: whether it means sp or the
    zero register depends on the position, so holes never take it."""

    __slots__ = ()

    def apply(self, buf: bytearray, at: int, target: int, place: int) -> None:
        if target == self.forbid:
            raise LinkError(f"{self.name}: register 31 (sp, xzr or wzr) cannot fill a register hole")
        super().apply(buf, at, target, place)


_reg_fields: dict[tuple, RegField] = {}


def reg_field(name: str, shift: int, width: int, take: int, gp: bool) -> RegField:
    """The (shared) RegField with these parameters."""
    key = (name, shift, width, take, False, 31 if gp else None)
    kind = _reg_fields.get(key)
    if kind is None:
        kind = _reg_fields[key] = RegField(*key)
    return kind


REL26 = BranchKind("rel26", 26, 0)
REL19 = BranchKind("rel19", 19, 5)
REL14 = BranchKind("rel14", 14, 5)
REL21_ADR = AdrKind("rel21_adr", page=False)
REL21_ADRP = AdrKind("rel21_adrp", page=True)

__all__ = ["BranchKind", "AdrKind", "RegField", "reg_field", "REL26", "REL19", "REL14", "REL21_ADR", "REL21_ADRP"]
