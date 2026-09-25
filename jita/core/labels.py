"""Labels, indexed PC labels and external symbols."""

import itertools
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .assembler import Assembler
    from .section import Section

# Global bind order. Images remember the value at link time, so labels bound
# afterwards (even at the very end of a section) can be told apart.
_bind_seq = itertools.count(1)


class Label:
    """A position in a section. Hashable by identity, never by name.

    Named labels become exported symbols of the linked image. A label is
    defined with `label(lbl)` inside an assembler context or `a.label(lbl)`.
    A label with an `owner` can only be bound into that assembler.
    """

    __slots__ = ("name", "section", "offset", "owner", "_pc", "_seq")

    def __init__(self, name: str | None = None, *, owner: Assembler | None = None):
        self.name = name
        self.owner = owner
        self.section: Section | None = None
        self.offset: int | None = None
        self._pc: int | None = None  # index when created by PcLabels
        self._seq = 0  # position in the global bind order, 0 while unbound

    @property
    def bound(self) -> bool:
        return self.section is not None

    def __str__(self) -> str:
        if self.name is not None:
            return self.name
        if self._pc is not None:
            return f".Lpc{self._pc}"
        return f".L{id(self):x}"

    def __repr__(self) -> str:
        if self.name is not None:
            desc = repr(self.name)
        elif self._pc is not None:
            desc = f"pc[{self._pc}]"
        else:
            desc = f"<anon {id(self):#x}>"
        if self.section is not None:
            desc += f" @ {self.section.name}+{self.offset:#x}"
        return f"Label({desc})"


class PcLabels:
    """Indexed labels, DynASM `=>n`.

    `pc[i]` returns the same Label for the same i, creating it on first use.
    `len(pc)` is the max index + 1. Labels created here are owned by
    `owner` and are defined with `a.label(a.pc[i])`.
    """

    __slots__ = ("_labels", "owner")

    def __init__(self, owner: Assembler | None = None):
        self._labels: dict[int, Label] = {}
        self.owner = owner

    def __getitem__(self, i: int) -> Label:
        if not isinstance(i, int) or i < 0:
            raise IndexError(f"pc label index must be a non-negative int, got {i!r}")
        lbl = self._labels.get(i)
        if lbl is None:
            lbl = self._labels[i] = Label(owner=self.owner)
            lbl._pc = i
        return lbl

    def __len__(self) -> int:
        return max(self._labels, default=-1) + 1

    def __contains__(self, i: int) -> bool:
        return i in self._labels


class Extern:
    """External symbol.

    The address is supplied at link/load time via the `externs` mapping, or
    fixed up front with `Extern("memcpy", addr)`. The mapping wins if both
    are given.
    """

    __slots__ = ("name", "address")

    def __init__(self, name: str, address: int | None = None):
        self.name = name
        self.address = address

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        if self.address is None:
            return f"Extern({self.name!r})"
        return f"Extern({self.name!r}, {self.address:#x})"
