"""Fragments: code encoded once with holes, then instantiated many times.

This is DynASM's split between encoding (the preprocessor) and filling in
runtime values (`dasm_put`). A Fragment is encoded like any assembler, with
`Hole` operands standing for registers, immediates and labels. Every
instance copies the fragment's bytes, patches the holes' bits and fields,
and appends the result to another assembler; the encoder does not run again.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass, replace
from typing import Any

from .assembler import Assembler, _check_pow2, current
from .errors import EncodeError, JitaError, LinkError
from .labels import Extern, Label
from .operand import Hole, Imm, Register
from .patch import Patch, PatchKind
from .section import Section

__all__ = ["Fragment", "Instance"]


@dataclass(slots=True, eq=False)
class Instance:
    """One instantiation of a Fragment.

    `labels` maps the fragment's local labels to the fresh labels of this
    instance: named ones by name, anonymous ones by the original Label
    object. Label holes map from the Hole to the Label they were given.
    `values` holds the hole values by name (label names resolved to Labels).
    """

    fragment: Fragment
    section: Section
    start: int
    labels: dict[str | Hole | Label, Label] = field(default_factory=dict)
    values: dict[str, Any] = field(default_factory=dict)

    @property
    def end(self) -> int:
        return self.start + len(self.fragment)


class _Plan:
    """What instantiate needs, derived from the fragment's section once."""

    __slots__ = ("key", "data", "fields", "events", "insns", "local")

    def __init__(self, frag: Fragment):
        sec = frag.cur
        self.key = frag._version()
        self.data = bytes(sec.buf)
        # Register and immediate hole fields, applied to the copied bytes.
        self.fields: list[tuple[int, PatchKind, Hole]] = []
        relocs = []
        for p in sec.patches:
            t = p.target
            if isinstance(t, Hole) and t.kind != "label":
                self.fields.append((p.offset, p.kind, t))
            else:
                relocs.append((p.offset, 1, (p.kind, t, p.addend)))
        # Labels bound inside the fragment are local to each instance.
        self.local: list[Label] = list(frag.labels)
        binds = [(lbl.offset, 0, lbl) for lbl in self.local]
        # Binds come before a patch at the same offset; sort is stable.
        self.events = sorted(binds + relocs, key=lambda e: (e[0], e[1]))
        self.insns = list(sec.insns)


class Fragment(Assembler):
    """An Assembler with one section whose code is instantiated elsewhere.

    Encode into it as usual (`with frag:` or `frag.mov(...)`), using `Hole`
    operands for what changes between instances, then call
    `frag.instantiate(asm, name=value, ...)`. Labels defined inside the
    fragment are local to each instance. `align` is the alignment the
    instantiation position must have; `align(n)` inside the fragment is
    relative to its start and raises that requirement to n.
    """

    def __init__(self, arch: Any = None, *, align: int = 1):
        super().__init__(arch)
        _check_pow2(align)
        self.alignment = align
        self.holes: dict[str, Hole] = {}
        self._plan: _Plan | None = None

    def __repr__(self) -> str:
        return f"<Fragment {self.arch.name} {len(self)} bytes, holes {list(self.holes)}>"

    def __len__(self) -> int:
        return len(self.cur.buf)

    def __bool__(self) -> bool:
        # `asm or current()` must not skip an empty fragment.
        return True

    # restrictions

    def section(self, name: str, align: int | None = None, writable: bool | None = None):
        raise JitaError("a Fragment has exactly one section")

    def link(self, base: int = 0, externs=None):
        raise JitaError("a Fragment cannot be linked, instantiate it into an Assembler")

    def load(self, externs=None):
        raise JitaError("a Fragment cannot be loaded, instantiate it into an Assembler")

    # holes

    def accept_hole(self, hole: Hole) -> None:
        other = self.holes.get(hole.name)
        if other is None:
            self.holes[hole.name] = hole
        elif other is not hole:
            raise EncodeError(f"two different holes are named {hole.name!r} in this fragment")

    def declare(self, *holes: Hole) -> None:
        """Make holes known without using them, so instantiate accepts (and
        ignores) their values. Useful when the code emitted depends on
        Python conditions."""
        for h in holes:
            if not isinstance(h, Hole):
                raise TypeError(f"expected a Hole, got {h!r}")
            self.accept_hole(h)

    def emit_patch(self, kind: PatchKind, target: Label | Extern | Hole, addend: int = 0) -> None:
        # A kind wrapping another one with an Extern target is a request for
        # an extern pointer slot. The fragment keeps it unchanged; the
        # assembler it is instantiated into creates the slot.
        if getattr(kind, "field", None) is not None and isinstance(target, Extern):
            sec = self.cur
            sec.patches.append(Patch(sec.pos(), kind, target, addend))
            sec.buf += bytes(kind.size)
            return
        super().emit_patch(kind, target, addend)

    def align(self, n: int, fill: bytes | None = None) -> None:
        super().align(n, fill)
        self.alignment = max(self.alignment, n)

    # instantiation

    def _version(self) -> tuple[int, ...]:
        sec = self.cur
        return (len(sec.buf), len(sec.patches), len(self.labels), len(sec.insns))

    def _compiled(self) -> _Plan:
        plan = self._plan
        if plan is None or plan.key != self._version():
            plan = self._plan = _Plan(self)
        return plan

    def _value(self, asm: Assembler, hole: Hole, v: Any) -> Any:
        if hole.kind == "reg":
            if (
                isinstance(v, Register)
                and v.kind == hole.regclass
                and v.size == hole.size
                and not getattr(v, "high", False)
            ):
                return v
            what = "an 8 bit register other than ah/ch/dh/bh" if hole.size == 1 else "a register"
            raise EncodeError(f"{hole!r} needs {what} of its class and size, got {v!r}")
        if hole.kind == "imm":
            if isinstance(v, Imm):
                v = v.value
            if not isinstance(v, int) or isinstance(v, bool):
                raise EncodeError(f"{hole!r} needs an int, got {v!r}")
            lo, hi = hole.range
            if not lo <= v <= hi:
                raise EncodeError(f"{hole!r}: value {v} out of range {lo}..{hi}")
            return v
        if isinstance(v, Label):
            if v.owner is not None and v.owner is not asm:
                raise LinkError(f"{hole!r}: {v!r} belongs to another assembler")
            return v
        if isinstance(v, str) and v:
            return v  # resolved with asm.named once every value is checked
        raise EncodeError(f"{hole!r} needs a Label or a label name, got {v!r}")

    def _outside(self, asm: Assembler, lbl: Label) -> Label:
        """The label of `asm` that a label not bound in the fragment means."""
        if lbl.owner is self:
            if lbl.name is not None:
                return asm.named(lbl.name)
            if lbl._pc is not None:
                return asm.pc[lbl._pc]
            raise LinkError(f"{lbl!r} is never bound in the fragment")
        if lbl.owner is not None and lbl.owner is not asm:
            raise LinkError(f"{lbl!r} belongs to another assembler")
        return lbl

    def instantiate(self, asm: Assembler | None = None, /, **values: Any) -> Instance:
        """Emit a copy of the fragment into `asm` (default: the current
        assembler) with every hole replaced by the value given for its name:
        a register of the hole's class and size, an int in range, or a
        Label or label name. Returns the Instance."""
        if asm is None:
            asm = current()
        if asm is self:
            raise JitaError("a Fragment cannot be instantiated into itself")
        if asm.arch.name != self.arch.name:
            raise EncodeError(f"{self!r} is {self.arch.name} code, {asm!r} is not")
        missing = [n for n in self.holes if n not in values]
        if missing:
            raise TypeError(f"instantiate: missing value for hole {', '.join(map(repr, missing))}")
        extra = [n for n in values if n not in self.holes]
        if extra:
            raise TypeError(f"instantiate: no hole named {', '.join(map(repr, extra))}")
        vals = {h: self._value(asm, h, values[n]) for n, h in self.holes.items()}
        plan = self._compiled()

        # Everything that can fail happens before anything is emitted.
        buf = bytearray(plan.data)
        for off, kind, hole in plan.fields:
            v = vals[hole]
            try:
                kind.apply(buf, off, v.code if hole.kind == "reg" else v, 0)
            except LinkError as e:
                raise EncodeError(f"{hole!r} = {v!r}: {e}") from None
        start = asm.pos()
        if start % self.alignment:
            raise LinkError(
                f"{self!r} needs a position aligned to {self.alignment}, "
                f"{asm.cur.name}+{start:#x} is not"
            )
        for h, v in vals.items():
            if isinstance(v, str):
                vals[h] = asm.named(v)
        fresh = {lbl: Label(owner=asm) for lbl in plan.local}
        targets: dict[Any, Any] = {}

        def target(t: Any) -> Any:
            r = targets.get(t)
            if r is None:
                if isinstance(t, Hole):
                    r = vals[t]
                elif isinstance(t, Label):
                    r = fresh.get(t) or self._outside(asm, t)
                else:
                    r = t
                targets[t] = r
            return r

        for _, order, item in plan.events:
            if order:
                target(item[1])

        # Emit bytes, bind the fresh labels and re-emit label patches through
        # emit_patch, in offset order.
        sec = asm.cur
        sec.align = max(sec.align, self.alignment)
        pos = 0
        for off, order, item in plan.events:
            if off > pos:
                asm.emit(buf[pos:off])
                pos = off
            if order:
                kind, t, addend = item
                asm.emit_patch(kind, targets[t], addend)
                pos = off + kind.size
            else:
                asm.bind(fresh[item])
        if pos < len(buf):
            asm.emit(buf[pos:])

        def subst(op: Any) -> Any:
            if isinstance(op, (Hole, Label)):
                return target(op)
            if is_dataclass(op) and not isinstance(op, type):
                changes = {}
                for f in fields(op):
                    x = getattr(op, f.name)
                    if isinstance(x, (Hole, Label)):
                        changes[f.name] = target(x)
                return replace(op, **changes) if changes else op
            return op

        for lo, hi, mnemonic, ops in plan.insns:
            sec.insns.append((start + lo, start + hi, mnemonic, tuple(subst(op) for op in ops)))

        labels: dict[str | Hole | Label, Label] = {}
        for lbl in plan.local:
            labels[lbl.name if lbl.name is not None else lbl] = fresh[lbl]
        for h, v in vals.items():
            if h.kind == "label":
                labels[h] = v
        named = {n: vals[h] for n, h in self.holes.items() if n in values}
        return Instance(self, sec, start, labels, named)
