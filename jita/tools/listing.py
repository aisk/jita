"""Human readable dump of an assembler's sections."""

from typing import TYPE_CHECKING, Any

from ..core.labels import Extern, Label

if TYPE_CHECKING:
    from ..core.assembler import Assembler
    from ..core.link import Image
    from ..core.patch import Patch


class _Names:
    """Label names for one listing. Extern pointer slots are shown as
    `name@slot`; other anonymous labels are numbered .L1, .L2, ... in bind
    order, so the output does not depend on object ids."""

    def __init__(self, labels: list[Label], slots: dict[str, Label] | None = None):
        self._anon: dict[int, str] = {}
        self._slots = 0
        for name, lbl in (slots or {}).items():
            self._anon[id(lbl)] = f"{name}@slot"
            self._slots += 1
        for lbl in labels:
            self(lbl)

    def __call__(self, target: Any) -> str:
        if not isinstance(target, Label) or target.name is not None or target._pc is not None:
            return str(target)
        name = self._anon.get(id(target))
        if name is None:
            name = self._anon[id(target)] = f".L{len(self._anon) - self._slots + 1}"
        return name

    def operand(self, op: Any) -> str:
        if isinstance(op, Label):
            return self(op)
        if isinstance(op, int) and not isinstance(op, bool):
            return str(op) if -256 < op < 256 else hex(op)
        label = getattr(op, "label", None)  # rip-relative memory operand
        text = str(op)
        if isinstance(label, Label):
            text = text.replace(str(label), self(label))
        elif isinstance(label, Extern):
            # [rip + ext] reads the extern's pointer slot, not the extern.
            text = text.replace(f"rip+{label}", f"rip+{label.name}@slot")
        return text

    def patch(self, p: Patch) -> str:
        note = f"{p.kind.name} -> {self(p.target)}"
        if p.addend:
            note += f"{p.addend:+d}"
        return note


def listing(asm: Assembler, image: Image | None = None, width: int = 8) -> str:
    """Return a listing of every section: offsets, hex bytes, bound labels
    and patch annotations. With `image`, show addresses and linked bytes.

    Instructions are printed one per line with their operands and patches.
    Data is printed `width` bytes per row. Instructions longer than `width`
    bytes continue on the following rows."""
    names = _Names(asm.labels, getattr(asm, "extern_slots", None))
    labels: dict[int, dict[int, list[Label]]] = {}
    for lbl in asm.labels:
        if lbl.offset is not None:  # asm.labels holds bound labels only
            labels.setdefault(id(lbl.section), {}).setdefault(lbl.offset, []).append(lbl)

    lines = []
    for sec in asm.sections.values():
        head = f"section {sec.name} (align {sec.align}, {len(sec.buf)} bytes)"
        data: bytes | bytearray
        if image is not None:
            start = image.section_offsets[sec.name]
            data = image.data[start : start + len(sec.buf)]
            head += f" @ {image.base + start:#x}"
        else:
            start, data = 0, sec.buf
        lines.append(head)
        sec_labels = labels.get(id(sec), {})
        patches: dict[int, list[Patch]] = {}
        for p in sec.patches:
            patches.setdefault(p.offset, []).append(p)
        insns = {lo: (hi, mn, ops) for lo, hi, mn, ops in sec.insns if hi > lo}

        def row(off: int, end: int, text: str = "") -> None:
            addr = off if image is None else image.base + start + off
            lines.append(f"  {addr:08x}  {data[off:end].hex(' '):<{width * 3 + 1}}{text}".rstrip())

        # Chunk boundaries: labels, instructions, patch fields in data.
        cuts = {0, len(data)} | sec_labels.keys()
        inside: set[int] = set()
        for lo, (hi, _, _) in insns.items():
            cuts |= {lo, hi}
            inside.update(range(lo + 1, hi))
        for p in sec.patches:
            cuts |= {p.offset, p.offset + p.kind.size}
        bounds = sorted(c for c in cuts - inside if c <= len(data))
        ends: list[int | None] = [*bounds[1:], None]
        for lo, end in zip(bounds, ends):
            for lbl in sec_labels.get(lo, ()):
                lines.append(f"{names(lbl)}:")
            if end is None:
                break
            hi = end
            insn = insns.get(lo)
            if insn is not None and insn[0] == hi:
                _, mn, ops = insn
                text = f"{mn} {', '.join(map(names.operand, ops))}" if ops else mn
                notes = [names.patch(p) for off in range(lo, hi) for p in patches.get(off, ())]
                if notes:
                    text = f"{text:<32}; {', '.join(notes)}"
                row(lo, min(lo + width, hi), text)
                for off in range(lo + width, hi, width):
                    row(off, min(off + width, hi))
                continue
            for off in range(lo, hi, width):
                ps = patches.get(off)
                row(off, min(off + width, hi), "; " + ", ".join(map(names.patch, ps)) if ps else "")
    return "\n".join(lines)
