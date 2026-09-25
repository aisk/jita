"""The Assembler: sections, labels, data directives and the current context."""

from __future__ import annotations

import importlib
import platform
from collections.abc import Callable, Mapping
from contextvars import ContextVar, Token
from typing import TYPE_CHECKING, Any

from .arch import Arch
from .errors import EncodeError, JitaError, LinkError
from .labels import Extern, Label, PcLabels, _bind_seq
from .patch import ABS64, ABS_BY_SIZE, Patch, PatchKind, SlotKind
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


def label(target: str | Label | None = None) -> Label:
    """Define a label in the current assembler, see `Assembler.label`."""
    return current().label(target)


def _host_arch() -> Arch:
    machine = platform.machine().lower()
    if machine in ("x86_64", "amd64"):
        from .. import x64

        return x64.ARCH
    if machine in ("aarch64", "arm64"):
        from .. import aarch64

        return aarch64.ARCH
    raise JitaError(f"no jita architecture for host machine {machine!r}")


_ARCH_NAMES = ("x64", "aarch64")


def _resolve_arch(arch: Any) -> Arch:
    if arch is None:
        return _host_arch()
    if isinstance(arch, str):
        if arch not in _ARCH_NAMES:
            raise ValueError(f"unknown architecture {arch!r} (known: {', '.join(_ARCH_NAMES)})")
        arch = importlib.import_module(f"jita.{arch}")
    arch = getattr(arch, "ARCH", arch)
    if not isinstance(arch, Arch):
        raise TypeError(f"expected an Arch, a module with ARCH or an architecture name, got {arch!r}")
    return arch


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

    def __call__(self, *ops: Any, **kw: Any) -> Any:
        return self._fn(*ops, asm=self._asm, **kw)

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
        """`arch` is an Arch object, a package exposing one as `ARCH`
        (e.g. `jita.x64`) or the name of such a jita package ("x64",
        "aarch64"). None selects the host architecture."""
        self.arch: Arch = _resolve_arch(arch)
        self.sections: dict[str, Section] = {}
        self.cur: Section = self._get_section("code")
        self.pc = PcLabels(self)
        self.labels: list[Label] = []  # bound labels, in bind order
        self._symbols: dict[str, Label] = {}
        self._named: dict[str, Label] = {}  # referenced by name, not bound yet
        self.extern_slots: dict[str, Label] = {}  # extern name -> pointer slot

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

    def emit_patch(self, kind: PatchKind, target: Label | Extern, addend: int = 0) -> None:
        """Reserve kind.size zero bytes at pos and record the Patch over
        them with `add_patch`. Nothing stays emitted if the patch is
        rejected."""
        sec = self.cur
        start = sec.pos()
        sec.buf += bytes(kind.size)
        try:
            self.add_patch(start, kind, target, addend)
        except BaseException:
            del sec.buf[start:]
            raise

    def add_patch(self, offset: int, kind: PatchKind, target: Label | Extern, addend: int = 0) -> None:
        """Record a patch over bytes already emitted in the current section,
        without reserving new ones. The bytes there are the template the
        kind writes into: zero for x86 style fields or an instruction word
        whose field a kind ORs in (aarch64).

        A `SlotKind` patch to an Extern (`qword[rip + ext]`) is recorded as
        `kind.field` to the extern's pointer slot, see `extern_slot`.
        """
        if not isinstance(target, (Label, Extern)):
            raise TypeError(f"patch target must be a Label or Extern, got {target!r}")
        if isinstance(kind, SlotKind) and not isinstance(target, Extern):
            raise TypeError(f"{kind.name} patch target must be an Extern, got {target!r}")
        sec = self.cur
        if not 0 <= offset <= sec.pos() - kind.size:
            raise ValueError(f"patch at {offset} is outside the emitted bytes")
        if isinstance(kind, SlotKind):
            kind, target = kind.field, self.extern_slot(target)
        sec.patches.append(Patch(offset, kind, target, addend))

    def note_insn(self, start: int, mnemonic: str, ops: tuple = ()) -> None:
        """Record that the bytes from `start` to pos form one instruction.
        Encoders call this after emitting; listings use it to print one
        instruction per line."""
        sec = self.cur
        sec.insns.append((start, sec.pos(), mnemonic, tuple(ops)))

    def extern_slot(self, extern: Extern) -> Label:
        """The label of an 8 byte slot holding `extern`'s address.

        The first request for a name appends `qword(extern)` (an ABS64
        patch) to the read-only `externs` section, creating it with
        alignment 8 if needed. Later requests for the same name return the
        same label, so there is one slot per name per assembler; the slot's
        address comes from the first Extern object seen for that name
        (`externs` given to `link`/`load` override it as usual).
        """
        if not isinstance(extern, Extern):
            raise TypeError(f"extern_slot expects an Extern, got {extern!r}")
        slot = self.extern_slots.get(extern.name)
        if slot is not None:
            return slot
        sec = self.sections.get("externs")
        if sec is None:
            sec = self.sections["externs"] = Section("externs", align=8)
        prev, self.cur = self.cur, sec
        try:
            if pad := -sec.pos() % 8:
                self.emit(bytes(pad))
            slot = self.bind(Label(owner=self))
            self.emit_patch(ABS64, extern)
        finally:
            self.cur = prev
        self.extern_slots[extern.name] = slot
        return slot

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
                    f"define it with label({label.name!r})"
                )
            self._symbols[label.name] = label
        label.section, label.offset = self.cur, self.cur.pos()
        label._seq = next(_bind_seq)
        self.labels.append(label)
        return label

    def label(self, target: str | Label | None = None) -> Label:
        """Define a label at the current position, the `name:` line of an
        assembly file. `target` is a Label object, the name of a label (one
        referenced earlier as a string binds that same label) or None for a
        new anonymous label. Returns the bound label."""
        if target is None:
            return self.bind(Label(owner=self))
        if isinstance(target, Label):
            return self.bind(target)
        return self.bind(self.named(target))

    def named(self, name: str) -> Label:
        """The label called `name`, created unbound if it does not exist yet.

        This is what a string operand means: `jz("done")` is
        `jz(a.named("done"))`. The label is bound later by `label("done")`.
        Linking fails if it never is.
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

    def function(
        self,
        restype: Any,
        *argtypes: Any,
        entry: Label | str | None = None,
        externs: Mapping[str, int] | None = None,
    ) -> Any:
        """Load into a fresh Module and return a ctypes callable bound to
        `entry` (default: image base). The callable keeps the Module alive
        through its `module` attribute and this assembler through
        `assembler`; the memory is released when it is garbage collected."""
        from ..runtime.loader import load

        fn = load(self, externs).function(restype, *argtypes, entry=entry)
        fn.assembler = self
        return fn


def _check_pow2(n: int) -> None:
    if n <= 0 or n & (n - 1):
        raise ValueError(f"alignment must be a power of two, got {n}")
