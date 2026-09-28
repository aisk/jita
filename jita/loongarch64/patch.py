"""loongarch64 relocation kinds.

The encoder emits instruction words with the offset fields left zero and
the kind ORs the resolved field into them at link time. `place` is the
address of the (first) instruction; displacements are `target - place`.
Out of range or misaligned displacements raise LinkError.

    B16      beq, bne, blt, bge, bltu, bgeu    offs16 at bit 10, +-128KB
    B21      beqz, bnez, bceqz, bcnez          offs21, +-4MB
    B26      b, bl                             offs26, +-128MB
    PCREL20  pcaddi                            si20 at bit 5, +-2MB
    PCALA    pcalau12i + 12 bit immediate      page and low 12 bits, +-2GB
    CALL36   pcaddu18i + jirl                  +-128GB

The branch offsets count words. B21 and B26 split them: the low 16 bits
at bit 10, the rest at bit 0.

PCALA covers two words, `pcalau12i rX, hi20` and an instruction with a
signed 12 bit immediate at bit 10 (`addi.d`, a load or a store) that adds
the low 12 bits of the target. pcalau12i works on 4KB pages, like
aarch64's adrp: it adds `hi20 << 12` to the pc with its low 12 bits
cleared. The low 12 bits are sign extended by the second instruction, so
hi20 is `((target + 0x800) >> 12) - (place >> 12)`, as GNU ld computes
R_LARCH_PCALA_HI20.

CALL36 covers `pcaddu18i rX, hi20` and `jirl rd, rX, offs16`:
hi20 = `(n + 0x20000) >> 18` and offs16 is the rest in words, as for
R_LARCH_CALL36.

Absolute data (`a.dword(label)`) uses the core ABS kinds.
"""

from ..core.errors import LinkError
from ..core.patch import PatchKind


def _or_word(buf: bytearray, at: int, bits: int) -> None:
    word = int.from_bytes(buf[at : at + 4], "little") | bits
    buf[at : at + 4] = word.to_bytes(4, "little")


class BranchKind(PatchKind):
    """A pc relative word offset of `bits` bits. `split` is the number of
    bits at bit 10 (the rest go to bit 0); None puts the whole offset at
    `shift`."""

    __slots__ = ("bits", "split", "shift")

    def __init__(self, name: str, bits: int, split: int | None, shift: int = 10):
        super().__init__(name, 4)
        self.bits, self.split, self.shift = bits, split, shift

    def apply(self, buf: bytearray, at: int, target: int, place: int) -> None:
        n = target - place
        if n & 3:
            raise LinkError(f"{self.name}: displacement {n} is not a multiple of 4")
        limit = 1 << (self.bits + 1)
        if not -limit <= n < limit:
            raise LinkError(f"{self.name}: displacement {n} out of range")
        offs = (n >> 2) & ((1 << self.bits) - 1)
        if self.split is None:
            field = offs << self.shift
        else:
            field = (offs & 0xFFFF) << 10 | offs >> 16
        _or_word(buf, at, field)


def split_pcala(target: int, place: int) -> tuple[int, int] | None:
    """(hi20, lo12) for pcalau12i at `place` and the instruction after it,
    or None when hi20 does not fit 20 signed bits. lo12 is the signed
    value the second instruction adds."""
    hi = ((target + 0x800) >> 12) - (place >> 12)
    if not -(1 << 19) <= hi < (1 << 19):
        return None
    return hi, ((target & 0xFFF) ^ 0x800) - 0x800


def split_call36(n: int) -> tuple[int, int] | None:
    """(hi20, offs16) of a pcaddu18i + jirl displacement: n = hi20 << 18
    plus offs16 << 2. None when hi20 does not fit 20 signed bits."""
    hi = (n + 0x20000) >> 18
    if not -(1 << 19) <= hi < (1 << 19):
        return None
    return hi, (n - (hi << 18)) >> 2


class PcalaKind(PatchKind):
    """pcalau12i and the instruction after it, 8 bytes."""

    __slots__ = ()

    def __init__(self, name: str):
        super().__init__(name, 8)

    def apply(self, buf: bytearray, at: int, target: int, place: int) -> None:
        parts = split_pcala(target, place)
        if parts is None:
            raise LinkError(f"{self.name}: target {target:#x} out of range of {place:#x}")
        hi, lo = parts
        _or_word(buf, at, (hi & 0xFFFFF) << 5)
        _or_word(buf, at + 4, (lo & 0xFFF) << 10)


class Call36Kind(PatchKind):
    """pcaddu18i and the jirl after it, 8 bytes."""

    __slots__ = ()

    def __init__(self, name: str):
        super().__init__(name, 8)

    def apply(self, buf: bytearray, at: int, target: int, place: int) -> None:
        n = target - place
        if n & 3:
            raise LinkError(f"{self.name}: displacement {n} is not a multiple of 4")
        parts = split_call36(n)
        if parts is None:
            raise LinkError(f"{self.name}: displacement {n} out of range")
        hi, lo = parts
        _or_word(buf, at, (hi & 0xFFFFF) << 5)
        _or_word(buf, at + 4, (lo & 0xFFFF) << 10)


B16 = BranchKind("b16", 16, None)
B21 = BranchKind("b21", 21, 16)
B26 = BranchKind("b26", 26, 16)
PCREL20 = BranchKind("pcrel20", 20, None, 5)
PCALA = PcalaKind("pcala")
CALL36 = Call36Kind("call36")

__all__ = [
    "BranchKind", "PcalaKind", "Call36Kind", "B16", "B21", "B26", "PCREL20", "PCALA", "CALL36",
    "split_pcala", "split_call36",
]  # fmt: skip
