"""Section layout and patch application."""

from __future__ import annotations

import mmap
import weakref
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, NamedTuple

from .errors import LinkError
from .label import Extern, Label, _bind_seq

if TYPE_CHECKING:
    from .assembler import Assembler
    from .section import Section


@dataclass(slots=True)
class Image:
    base: int
    data: bytes  # whole image, sections concatenated with alignment padding
    section_offsets: dict[str, int]
    symbols: dict[str, int]  # named labels -> offset from base
    # Bytes at the start of the image that hold only read-only sections,
    # rounded up to page_size. Everything after it is writable.
    exec_size: int = 0
    page_size: int = mmap.PAGESIZE
    # Section name -> weak reference to the Section object, so labels bound
    # in another assembler's section of the same name are rejected.
    sections: dict[str, weakref.ref[Section]] = field(default_factory=dict, repr=False)
    # Section name -> size in bytes at link time.
    section_sizes: dict[str, int] = field(default_factory=dict)
    # Last value of the global label bind counter at link time. Labels
    # bound later are not part of this image.
    bind_seq: int = 0

    def address(self, label: Label | str) -> int:
        if isinstance(label, str):
            try:
                return self.base + self.symbols[label]
            except KeyError:
                raise LinkError(f"no symbol named {label!r}") from None
        if not label.bound:
            raise LinkError(f"label {label} is never bound")
        name = label.section.name
        size = self.section_sizes.get(name)
        if size is None:
            raise LinkError(f"{label!r}: label bound after linking (section unknown to this image)")
        ref = self.sections.get(name)
        if ref is None or ref() is not label.section:
            raise LinkError(f"{label!r} is bound in a different assembler")
        # A label at the very end of a section (offset == size) is fine if
        # it was bound before linking, so the bind order decides that case.
        if label.offset > size or label._seq > self.bind_seq:
            raise LinkError(f"{label!r}: label bound after linking")
        return self.base + self.section_offsets[name] + label.offset


class Layout(NamedTuple):
    offsets: dict[str, int]  # section name -> offset from the image start
    size: int  # total image size
    exec_size: int  # page rounded size of the read-only prefix


def layout(asm: Assembler, page_size: int = mmap.PAGESIZE) -> Layout:
    """Place read-only sections first in creation order, then writable
    sections starting at the next page boundary."""
    offsets: dict[str, int] = {}

    def place(pos: int, writable: bool) -> int:
        for sec in asm.sections.values():
            if bool(sec.writable) == writable:
                pos += -pos % sec.align
                offsets[sec.name] = pos
                pos += len(sec.buf)
        return pos

    pos = place(0, False)
    exec_size = pos + -pos % page_size
    if any(sec.writable for sec in asm.sections.values()):
        pos = place(exec_size, True)
    return Layout(offsets, pos, exec_size)


def max_align(asm: Assembler) -> int:
    return max(sec.align for sec in asm.sections.values())


def link(asm: Assembler, base: int, externs: Mapping[str, int] | None = None) -> Image:
    align = max_align(asm)
    if base % align:
        raise LinkError(f"base {base:#x} is not aligned to {align} (largest section alignment)")
    page_size = mmap.PAGESIZE
    offsets, size, exec_size = layout(asm, page_size)
    externs = externs or {}
    out = bytearray(size)
    for sec in asm.sections.values():
        start = offsets[sec.name]
        out[start : start + len(sec.buf)] = sec.buf

    for sec in asm.sections.values():
        start = offsets[sec.name]
        for p in sec.patches:
            where = f"{sec.name}+{p.offset:#x}"
            t = p.target
            if isinstance(t, Label):
                if not t.bound:
                    raise LinkError(f"label {t} is never bound (referenced at {where})")
                if asm.sections.get(t.section.name) is not t.section:
                    raise LinkError(f"{t!r} belongs to another assembler (referenced at {where})")
                target = base + offsets[t.section.name] + t.offset
            elif isinstance(t, Extern):
                target = externs.get(t.name, t.address)
                if target is None:
                    raise LinkError(f"extern {t.name} has no address (referenced at {where})")
            else:
                raise LinkError(f"unresolved {t!r} at {where}")
            try:
                p.kind.apply(out, start + p.offset, target + p.addend, base + start + p.offset)
            except LinkError as e:
                raise LinkError(f"{e} at {where} (target {t})") from None

    symbols = {
        lbl.name: offsets[lbl.section.name] + lbl.offset
        for lbl in asm.labels
        if lbl.name is not None
    }
    sections = {name: weakref.ref(sec) for name, sec in asm.sections.items()}
    sizes = {name: len(sec.buf) for name, sec in asm.sections.items()}
    return Image(
        base, bytes(out), offsets, symbols, exec_size, page_size, sections, sizes, _last_seq()
    )


def _last_seq() -> int:
    # Draw a value from the bind counter: every label bound so far has a
    # smaller one, every label bound later a larger one.
    return next(_bind_seq)
