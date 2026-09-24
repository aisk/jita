"""Output sections."""

from __future__ import annotations

from dataclasses import dataclass, field

from .patch import Patch


@dataclass(slots=True, eq=False, weakref_slot=True)
class Section:
    name: str
    buf: bytearray = field(default_factory=bytearray)
    patches: list[Patch] = field(default_factory=list)
    align: int = 16
    # Writable sections are placed after all read-only ones, starting on a
    # page boundary, and stay read-write when the image is loaded.
    writable: bool = False
    # Instruction spans recorded by encoders, in emission order:
    # (start, end, mnemonic, operands). Only used for listings.
    insns: list[tuple[int, int, str, tuple]] = field(default_factory=list)

    def pos(self) -> int:
        return len(self.buf)
