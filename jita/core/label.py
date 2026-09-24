"""Labels, indexed PC labels and external symbols."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .section import Section


class Label:
    """A position in a section. Hashable by identity, never by name.

    Named labels become exported symbols of the linked image.
    """

    __slots__ = ("name", "section", "offset", "_pc")

    def __init__(self, name: str | None = None):
        self.name = name
        self.section: Section | None = None
        self.offset: int | None = None
        self._pc: int | None = None  # index when created by PcLabels

    @property
    def bound(self) -> bool:
        return self.section is not None

    def here(self) -> Label:
        """Bind at the current position of the current assembler."""
        from .assembler import current

        return current().bind(self)

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
    `len(pc)` is the max index + 1.
    """

    __slots__ = ("_labels",)

    def __init__(self):
        self._labels: dict[int, Label] = {}

    def __getitem__(self, i: int) -> Label:
        if not isinstance(i, int) or i < 0:
            raise IndexError(f"pc label index must be a non-negative int, got {i!r}")
        lbl = self._labels.get(i)
        if lbl is None:
            lbl = self._labels[i] = Label()
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
