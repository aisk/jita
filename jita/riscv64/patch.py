"""riscv64 relocation kinds.

The encoder emits instruction words with the immediate field left zero and
the kind ORs the resolved field into them at link time. `place` is the
address of the (first) instruction; displacements are `target - place`.
Out of range or misaligned displacements raise LinkError.

    B12       beq, bne, blt, bge, bltu, bgeu    B-type imm, +-4KB, even
    J20       jal, j                            J-type imm, +-1MB, even
    PCREL32   auipc + I or S-type instruction   hi20 and lo12, +-2GB

PCREL32 covers two words: `auipc rX, hi20` and the instruction that adds
the low 12 bits (`addi`, a load, `jalr` or a store), so the pair reaches
any byte within about +-2GB. hi20 is `(n + 0x800) >> 12`, rounded so that
the sign extended lo12 = `n - (hi20 << 12)` brings it back. The second
word's opcode tells an S-type store from the I-type forms.

Absolute data (`a.qword(label)`) uses the core ABS kinds.
"""

from ..core.errors import LinkError
from ..core.patch import PatchKind


def _or_word(buf: bytearray, at: int, bits: int) -> None:
    word = int.from_bytes(buf[at : at + 4], "little") | bits
    buf[at : at + 4] = word.to_bytes(4, "little")


def _b_type(n: int) -> int:
    return (
        ((n >> 12) & 1) << 31
        | ((n >> 5) & 0x3F) << 25
        | ((n >> 1) & 0xF) << 8
        | ((n >> 11) & 1) << 7
    )


def _j_type(n: int) -> int:
    return (
        ((n >> 20) & 1) << 31
        | ((n >> 1) & 0x3FF) << 21
        | ((n >> 11) & 1) << 20
        | ((n >> 12) & 0xFF) << 12
    )


def _s_type(n: int) -> int:
    return ((n >> 5) & 0x7F) << 25 | (n & 0x1F) << 7


def _i_type(n: int) -> int:
    return (n & 0xFFF) << 20


class BranchKind(PatchKind):
    """B-type or J-type branch offset: `bits` bit signed, even."""

    __slots__ = ("bits", "field")

    def __init__(self, name: str, bits: int):
        super().__init__(name, 4)
        self.bits = bits
        self.field = _b_type if bits == 13 else _j_type

    def apply(self, buf: bytearray, at: int, target: int, place: int) -> None:
        n = target - place
        if n & 1:
            raise LinkError(f"{self.name}: displacement {n} is not a multiple of 2")
        limit = 1 << (self.bits - 1)
        if not -limit <= n < limit:
            raise LinkError(f"{self.name}: displacement {n} out of range")
        _or_word(buf, at, self.field(n))


def split_pcrel(n: int) -> tuple[int, int] | None:
    """(hi20, lo12) of a pc-relative displacement for auipc and the
    instruction after it, or None when hi20 does not fit 20 signed bits."""
    hi = (n + 0x800) >> 12
    if not -(1 << 19) <= hi < (1 << 19):
        return None
    return hi, n - (hi << 12)


# Opcodes of the S-type instructions a PCREL32 pair can end with.
_STORES = frozenset([0x23, 0x27])


class PcrelKind(PatchKind):
    """auipc and the instruction after it, 8 bytes."""

    __slots__ = ()

    def __init__(self, name: str):
        super().__init__(name, 8)

    def apply(self, buf: bytearray, at: int, target: int, place: int) -> None:
        n = target - place
        parts = split_pcrel(n)
        if parts is None:
            raise LinkError(f"{self.name}: displacement {n} out of range")
        hi, lo = parts
        _or_word(buf, at, (hi & 0xFFFFF) << 12)
        second = buf[at + 4] & 0x7F
        _or_word(buf, at + 4, _s_type(lo) if second in _STORES else _i_type(lo))


B12 = BranchKind("b12", 13)
J20 = BranchKind("j20", 21)
PCREL32 = PcrelKind("pcrel32")

__all__ = ["BranchKind", "PcrelKind", "B12", "J20", "PCREL32", "split_pcrel"]
