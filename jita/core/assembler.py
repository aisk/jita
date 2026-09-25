"""The Assembler: sections, labels, data directives and the current context."""

from __future__ import annotations

import platform
from collections.abc import Callable, Mapping
from contextvars import ContextVar, Token
from typing import TYPE_CHECKING, Any

from .arch import Arch
from .errors import EncodeError, JitaError, LinkError
from .label import Extern, Label, PcLabels, _bind_seq
from .operand import Hole
from .patch import ABS_BY_SIZE, Patch, PatchKind
from .section import Section

if TYPE_CHECKING:
    from ..runtime.loader import Module
    from .link import Image

_current: ContextVar[Assembler | None] = ContextVar("jita_assembler", default=None)
# Tokens of the `with asm:` blocks active in this context, innermost last.
# Kept in a ContextVar so concurrent tasks entering the same Assembler each
# reset their own token.
_tokens: ContextVar[tuple[Token, ...]] = ContextVar("jita_assembler_tokens", default=())


def current() -> Assembler:
    asm = _current.get()
    if asm is None:
        raise JitaError("no active Assembler")
    return asm


def _host_arch() -> Arch:
    machine = platform.machine().lower()
    if machine in ("x86_64", "amd64"):
        from .. import x64

        return x64.ARCH
    raise JitaError(f"no jita architecture for host machine {machine!r}")


class _SectionSwitch:
    """Returned by Assembler.section. The switch has already happened; using
    it as a context manager restores the previous section on exit."""

    __slots__ = ("_asm", "_prev", "section")

    def __init__(self, asm: Assembler, prev: Section, section: Section):
        self._asm, self._prev, self.section = asm, prev, section

    def __enter__(self) -> Section:
        return self.section

    def __exit__(self, *exc) -> None:
        self._asm.cur = self._prev


class _BoundInsn:
    # A mnemonic function bound to an assembler. Callable attributes are
    # bound too, so `a.jmp.short(lbl)` works like `jmp.short(lbl)`; other
    # attributes (`__doc__`, `__name__`) come from the function unchanged.

    __slots__ = ("_fn", "_asm")

    def __init__(self, fn: Callable[..., Any], asm: Assembler):
        self._fn, self._asm = fn, asm

    def __call__(self, *ops: Any) -> Any:
        return self._fn(*ops, asm=self._asm)

    @property
    def __doc__(self) -> str | None:  # type: ignore[override]
        return self._fn.__doc__

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._fn, name)
        return _BoundInsn(attr, self._asm) if callable(attr) else attr

    def __repr__(self) -> str:
        return f"<bound {self._fn!r} of {self._asm!r}>"


class Assembler:
    def __init__(self, arch: Any = None):
        """`arch` is an Arch object, or a package exposing one as `ARCH`
        (e.g. `jita.x64`). None selects the host architecture."""
        if arch is None:
            arch = _host_arch()
        self.arch: Arch = getattr(arch, "ARCH", arch)
        self.sections: dict[str, Section] = {}
        self.cur: Section = self._get_section("code")
        self.pc = PcLabels(self)
        self.labels: list[Label] = []  # bound labels, in bind order
        self._symbols: dict[str, Label] = {}
        self._named: dict[str, Label] = {}  # referenced by name, not bound yet

    def __repr__(self) -> str:
        return f"<Assembler {self.arch.name} [{', '.join(self.sections)}]>"

    # context

    def __enter__(self) -> Assembler:
        _tokens.set((*_tokens.get(), _current.set(self)))
        return self

    def __exit__(self, *exc) -> None:
        stack = _tokens.get()
        if not stack:
            raise JitaError("Assembler context exited without a matching enter")
        _tokens.set(stack[:-1])
        _current.reset(stack[-1])

    # sections

    def _get_section(self, name: str) -> Section:
        sec = self.sections.get(name)
        if sec is None:
            sec = self.sections[name] = Section(name)
        return sec

    def section(
        self, name: str, align: int | None = None, writable: bool | None = None
    ) -> _SectionSwitch:
        """Switch to (creating if needed) a section. Usable both as a plain
        call that switches permanently and as `with a.section("data"):` which
        restores the previous section on exit. `align` raises the section's
        alignment in the linked image. `writable` sets whether the section
        stays read-write after loading; None keeps the current setting."""
        if align is not None:
            _check_pow2(align)
        prev = self.cur
        self.cur = sec = self._get_section(name)
        if align is not None:
            sec.align = max(sec.align, align)
        if writable is not None:
            sec.writable = bool(writable)
        return _SectionSwitch(self, prev, sec)

    # raw emission

    def pos(self) -> int:
        return self.cur.pos()

    def emit(self, data: bytes | bytearray) -> None:
        self.cur.buf += data

    def emit_patch(self, kind: PatchKind, target: Label | Extern | Hole, addend: int = 0) -> None:
        """Reserve kind.size zero bytes at pos and record the Patch."""
        if not isinstance(target, (Label, Extern, Hole)):
            raise TypeError(f"patch target must be a Label, Extern or Hole, got {target!r}")
        sec = self.cur
        sec.patches.append(Patch(sec.pos(), kind, target, addend))
        sec.buf += bytes(kind.size)

    def note_insn(self, start: int, mnemonic: str, ops: tuple = ()) -> None:
        """Record that the bytes from `start` to pos form one instruction.
        Encoders call this after emitting; listings use it to print one
        instruction per line."""
        sec = self.cur
        sec.insns.append((start, sec.pos(), mnemonic, tuple(ops)))

    # labels

    def bind(self, label: Label) -> Label:
        if label.bound:
            raise LinkError(f"{label!r} is already bound")
        if label.owner is not None and label.owner is not self:
            raise LinkError(f"{label!r} belongs to another assembler")
        if label.name is not None:
            if label.name in self._symbols:
                raise LinkError(f"duplicate label name {label.name!r}")
            forward = self._named.pop(label.name, None)
            if forward is not None and forward is not label:
                self._named[label.name] = forward
                raise LinkError(
                    f"label {label.name!r} is already referenced by name, "
                    f"bind it with a.label({label.name!r}) or a.named({label.name!r}).here()"
                )
            self._symbols[label.name] = label
        label.section, label.offset = self.cur, self.cur.pos()
        label._seq = next(_bind_seq)
        self.labels.append(label)
        return label

    def label(self, name: str | None = None) -> Label:
        """Create a label and bind it here. A name that was referenced
        earlier as a string binds that forward reference."""
        if name is not None and name in self._named:
            return self.bind(self._named[name])
        return self.bind(Label(name, owner=self))

    def named(self, name: str) -> Label:
        """The label called `name`, created unbound if it does not exist yet.

        This is what a string operand means: `jz("done")` is
        `jz(a.named("done"))`. The label is bound later by `a.label("done")`
        or `a.named("done").here()`. Linking fails if it never is.
        """
        if not isinstance(name, str) or not name:
            raise TypeError(f"label name must be a non-empty str, got {name!r}")
        lbl = self._symbols.get(name) or self._named.get(name)
        if lbl is None:
            lbl = self._named[name] = Label(name, owner=self)
        return lbl

    # data directives

    def _data(self, size: int, vals: tuple) -> None:
        for v in vals:
            if isinstance(v, str):
                v = self.named(v)
            if isinstance(v, (Label, Extern)):
                self.emit_patch(ABS_BY_SIZE[size], v)
                continue
            if not isinstance(v, int) or isinstance(v, bool):
                raise TypeError(f"data value must be an int, str, Label or Extern, got {v!r}")
            bits = size * 8
            if not -(1 << (bits - 1)) <= v < (1 << bits):
                raise EncodeError(f"value {v:#x} does not fit in {size} bytes")
            self.emit((v & ((1 << bits) - 1)).to_bytes(size, "little"))

    def byte(self, *vals: int | str | Label | Extern) -> None:
        self._data(1, vals)

    def word(self, *vals: int | str | Label | Extern) -> None:
        self._data(2, vals)

    def dword(self, *vals: int | str | Label | Extern) -> None:
        self._data(4, vals)

    def qword(self, *vals: int | str | Label | Extern) -> None:
        self._data(8, vals)

    def bytes(self, data: bytes) -> None:
        self.emit(data)

    def align(self, n: int, fill: bytes | None = None) -> None:
        """Pad to a multiple of n. `fill` is repeated; None uses the arch's
        NOP padding in sections that contain instructions and zero bytes
        in pure data sections. Also raises the section alignment to at
        least n."""
        _check_pow2(n)
        self.cur.align = max(self.cur.align, n)
        pad = -self.pos() % n
        if not pad:
            return
        if fill is None:
            self.emit(self.arch.nop_fill(pad) if self.cur.insns else bytes(pad))
        elif not fill:
            raise ValueError("align fill must not be empty")
        else:
            self.emit((fill * (pad // len(fill) + 1))[:pad])

    def space(self, n: int, fill: int = 0) -> None:
        if n < 0:
            raise ValueError(f"space size must not be negative, got {n}")
        self.emit(bytes([fill]) * n)

    # mnemonics as methods

    def __getattr__(self, name: str) -> _BoundInsn:
        if name.startswith("_"):
            raise AttributeError(name)
        fn = self.arch.insns.get(name)
        if fn is None:
            raise AttributeError(f"{self.arch.name} has no instruction {name!r}")
        return _BoundInsn(fn, self)

    # linking

    def link(self, base: int = 0, externs: Mapping[str, int] | None = None) -> Image:
        from .link import link

        return link(self, base, externs)

    def load(self, externs: Mapping[str, int] | None = None) -> Module:
        from ..runtime.loader import load

        return load(self, externs)


def _check_pow2(n: int) -> None:
    if n <= 0 or n & (n - 1):
        raise ValueError(f"alignment must be a power of two, got {n}")
