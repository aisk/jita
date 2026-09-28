"""riscv64 template interpreter: (mnemonic, operands) -> instruction words.

Works like `jita.aarch64.encoder`: the "|" separated alternatives of the
template (`jita.riscv64.table`) are tried in order and the first one that
accepts the operands wins. Every value is known when an instruction is
encoded, so immediates are checked and placed here, and only label
targets become patches (see `jita.riscv64.patch`).

Every instruction is one 32 bit word, except the pseudo instructions that
expand to several: `li` (1 to 8 words), and the auipc pairs of `lla`,
`la`, `call`, `tail` and loads and stores to a label. A pseudo
instruction is one entry in the listing, with all of its words.

Immediates are range checked by value and never truncated. The listing
shows operands as objdump does: I-type immediates in decimal, U-type
fields, shift amounts and unnamed CSR numbers in hex, `(a0)` for the
address of an atomic.
"""

from collections.abc import Sequence
from typing import Any

from ..core.errors import EncodeError
from ..core.labels import Extern, Label
from ..core.operand import Imm, Operand
from .mem import Addr, MemExpr
from .patch import B12, J20, PCREL32
from .regs import F, Reg, X
from .table import MAP_CSR, MAP_FENCE, MAP_OP, MAP_RM

__all__ = ["encode", "li_words", "MNEMONIC_ARGC", "RM_MNEMONICS"]


def _build_argc() -> dict[str, tuple[int, ...]]:
    out: dict[str, set[int]] = {}
    for key in MAP_OP:
        name, _, argc = key.rpartition("_")
        out.setdefault(name, set()).add(int(argc))
    return {k: tuple(sorted(v)) for k, v in out.items()}


# Mnemonic -> accepted operand counts.
MNEMONIC_ARGC: dict[str, tuple[int, ...]] = _build_argc()

# Mnemonics that take the `rm=` rounding mode keyword.
RM_MNEMONICS: frozenset[str] = frozenset(
    k.rpartition("_")[0] for k, t in MAP_OP.items() if any("r" in alt[8:] for alt in t.split("|"))
)

_CSR_NAMES = {v: k for k, v in MAP_CSR.items()}


class _Fail(Exception):
    """One template alternative rejected the operands. `rank` is 0 for an
    operand of the wrong kind and 1 for a bad value; `pos` is the operand
    index. The failure with the highest (rank, pos) is reported."""

    def __init__(self, msg: str, pos: int, rank: int = 0):
        super().__init__(msg)
        self.msg, self.pos, self.rank = msg, pos, rank


def _is_int(x: Any) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def _fmt(n: int) -> str:
    return str(n) if -256 < n < 256 else hex(n)


class _Hex(Imm):
    """An immediate that listings show in hex, as objdump does for U-type
    fields, shift amounts and CSR numbers."""

    __slots__ = ()

    def __str__(self) -> str:
        return hex(self.value)


class _Text(Operand):
    """A listing operand shown as given text (`(a0)` for atomics)."""

    __slots__ = ("text",)

    def __init__(self, text: str):
        self.text = text

    def __str__(self) -> str:
        return self.text


# -- li --------------------------------------------------------------------------


def _addi(rd: int, rs: int, imm: int, op: int = 0x13) -> int:
    return (imm & 0xFFF) << 20 | rs << 15 | rd << 7 | op


def _load_const(rd: int, v: int, out: list[int]) -> None:
    # GNU as' load_const: reduce to a 32 bit signed constant with slli and
    # addi, then lui and/or addiw.
    lower = ((v & 0xFFF) ^ 0x800) - 0x800
    upper = v - lower
    if not -(1 << 31) <= v < (1 << 31):
        upper = ((upper + (1 << 63)) & ((1 << 64) - 1)) - (1 << 63)  # int64 wrap
        shift = 12
        while not (upper >> shift) & 1:
            shift += 1
        _load_const(rd, upper >> shift, out)
        out.append(shift << 20 | rd << 15 | 1 << 12 | rd << 7 | 0x13)  # slli
        if lower:
            out.append(_addi(rd, rd, lower))
        return
    hi = 0
    if upper:
        out.append((upper >> 12 & 0xFFFFF) << 12 | rd << 7 | 0x37)  # lui
        hi = rd
    if lower or not hi:
        out.append(_addi(rd, hi, lower, 0x1B))  # addiw


def li_words(rd: int, value: int) -> list[int]:
    """The words of `li rd, value`: GNU as' expansion. `value` is any 64
    bit pattern, signed or unsigned."""
    v = ((value + (1 << 63)) & ((1 << 64) - 1)) - (1 << 63)
    if -2048 <= v < 2048:
        return [_addi(rd, 0, v)]  # addi rd, zero, v
    out: list[int] = []
    _load_const(rd, v, out)
    return out


# -- per alternative state ------------------------------------------------------


class _Alt:
    """Parses one template alternative."""

    def __init__(self, asm: Any, params: list[Any], rm: str | None):
        self.asm = asm
        self.params = params
        self.shown: list[Any] = list(params)  # operands as listings print them
        self.rm = rm
        self.n = 0
        self.target: Label | Extern | None = None
        self.branch: Any = None  # B12 or J20 for a branch, None otherwise
        self.pcrel = ""  # "G" or "g" for an auipc pair
        self.li: int | None = None

    def fail(self, msg: str, rank: int = 0, pos: int | None = None) -> _Fail:
        return _Fail(msg, self.n if pos is None else pos, rank)

    def param(self) -> Any:
        if self.n >= len(self.params):
            raise self.fail("missing operand")
        return self.params[self.n]

    def next(self) -> None:
        self.n += 1

    # registers

    def reg(self, cls: type[Reg], shift: int) -> int:
        q = self.param()
        if not isinstance(q, cls):
            what = "an integer register" if cls is X else "a floating point register"
            raise self.fail(f"expected {what}")
        self.next()
        return q.code << shift

    # immediates

    def int_param(self) -> int:
        q = self.param()
        if not _is_int(q):
            raise self.fail("expected an immediate")
        return q

    def imm(self, lo: int, hi: int, what: str, hexfmt: bool = False) -> int:
        n = self.int_param()
        if not lo <= n <= hi:
            raise self.fail(f"immediate {_fmt(n)} out of range ({what})", 1)
        self.shown[self.n] = _Hex(n) if hexfmt else Imm(n)
        self.next()
        return n

    def upper(self) -> int:
        n = self.int_param()
        if not -0x80000 <= n <= 0xFFFFF:
            raise self.fail(f"immediate {_fmt(n)} out of range (20 bit, 0..0xfffff or -0x80000..-1)", 1)
        self.shown[self.n] = _Hex(n & 0xFFFFF)
        self.next()
        return (n & 0xFFFFF) << 12

    def csr(self) -> int:
        q = self.param()
        if isinstance(q, str):
            n = MAP_CSR.get(q)
            if n is None:
                raise self.fail(f"unknown CSR name {q!r} (known: {', '.join(MAP_CSR)})", 1)
        elif _is_int(q):
            n = q
            if not 0 <= n <= 0xFFF:
                raise self.fail(f"CSR number {_fmt(n)} out of range (0..0xfff)", 1)
            name = _CSR_NAMES.get(n)
            self.shown[self.n] = _Hex(n) if name is None else name
        else:
            raise self.fail("expected a CSR number or name")
        self.next()
        return n << 20

    def fence_set(self, shift: int) -> int:
        q = self.param()
        if not isinstance(q, str):
            raise self.fail('expected a fence set such as "rw" or "iorw"')
        bits = 0
        last = 16
        for ch in q:
            b = MAP_FENCE.get(ch)
            if b is None or b >= last:
                raise self.fail(f"bad fence set {q!r} (letters of i, o, r, w in that order)", 1)
            bits |= b
            last = b
        if not bits:
            raise self.fail("empty fence set", 1)
        self.next()
        return bits << shift

    # addresses

    def mem(self, kind: str) -> int:
        m = self.param()
        if isinstance(m, Addr):
            raise self.fail(f"wrap the address in mem[...]: mem[{m}]", 1)
        if not isinstance(m, MemExpr):
            raise self.fail("expected a memory operand")
        n = m.disp
        op = m.base.code << 15
        if kind == "A":
            if n:
                raise self.fail(f"atomic address takes no offset, got {n}", 1)
            self.shown[self.n] = _Text(f"({m.base})")
        elif not -2048 <= n <= 2047:
            raise self.fail(f"offset {_fmt(n)} out of range (signed 12 bit, -2048..2047)", 1)
        elif kind == "S":
            op |= ((n >> 5) & 0x7F) << 25 | (n & 0x1F) << 7
        else:
            op |= (n & 0xFFF) << 20
        self.next()
        return op

    # labels

    def label(self) -> None:
        q = self.param()
        if isinstance(q, str):
            q = self.asm.named(q)
        elif not isinstance(q, (Label, Extern)):
            raise self.fail("expected a label")
        self.target = q
        self.next()


def _parse_template(t: str, alt: _Alt) -> int:
    op = int(t[:8], 16)
    for p in t[8:]:
        if p == "D":
            op |= alt.reg(X, 7)
        elif p == "N":
            op |= alt.reg(X, 15)
        elif p == "M":
            op |= alt.reg(X, 20)
        elif p == "d":
            op |= alt.reg(F, 7)
        elif p == "n":
            op |= alt.reg(F, 15)
        elif p == "m":
            op |= alt.reg(F, 20)
        elif p == "a":
            op |= alt.reg(F, 27)
        elif p == "E":
            op |= ((op >> 7) & 31) << 15
        elif p == "^":
            op |= ((op >> 15) & 31) << 20
        elif p == "I":
            op |= (alt.imm(-2048, 2047, "signed 12 bit, -2048..2047") & 0xFFF) << 20
        elif p == "U":
            op |= alt.upper()
        elif p == "H":
            op |= alt.imm(0, 63, "shift amount 0..63", True) << 20
        elif p == "W":
            op |= alt.imm(0, 31, "shift amount 0..31", True) << 20
        elif p == "C":
            op |= alt.csr()
        elif p == "K":
            op |= alt.imm(0, 31, "unsigned 5 bit, 0..31") << 15
        elif p in "LSA":
            op |= alt.mem(p)
        elif p == "B":
            alt.label()
            alt.branch = B12
        elif p == "J":
            alt.label()
            alt.branch = J20
        elif p == "G" or p == "g":
            alt.label()
            alt.pcrel = p
        elif p == "P":
            op |= alt.fence_set(24)
        elif p == "Q":
            op |= alt.fence_set(20)
        elif p == "#":
            n = alt.int_param()
            if not -(1 << 63) <= n < (1 << 64):
                raise alt.fail(f"immediate {_fmt(n)} out of range (64 bit)", 1)
            alt.li = n
            alt.next()
        elif p == "~":
            if (op >> 15) & 31 == (op >> 20) & 31:
                raise alt.fail("the auipc register must differ from the register stored", 1, len(alt.params) - 1)
        elif p == "r":
            rm = "dyn" if alt.rm is None else alt.rm
            v = MAP_RM.get(rm) if isinstance(rm, str) else None
            if v is None:
                raise alt.fail(f"unknown rounding mode {rm!r} (known: {', '.join(MAP_RM)})", 2)
            op |= v << 12
        else:
            raise AssertionError(f"bad template character {p!r} in {t!r}")
    if alt.n < len(alt.params):
        raise alt.fail("too many operands")
    if alt.pcrel == "G" and not (op >> 15) & 31:
        # auipc zero would drop the high part: GNU as rejects these too.
        raise alt.fail("zero cannot hold the auipc address of a label reference", 1, len(alt.params) - 1)
    return op


# -- entry point ---------------------------------------------------------------


def _flatten(ops: Sequence[Any]) -> list[Any]:
    return [op.value if isinstance(op, Imm) else op for op in ops]


def _error(mnemonic: str, ops: Sequence[Any], why: str | None = None) -> EncodeError:
    msg = f"{mnemonic}: no encoding for ({', '.join(repr(o) for o in ops)})"
    if why:
        msg += f": {why}"
    return EncodeError(msg)


def encode(asm: Any, mnemonic: str, ops: Sequence[Any], rm: str | None = None) -> None:
    """Encode one instruction (or pseudo instruction) and emit it into
    `asm`. `mnemonic` is the table name ("fadd.d", "and"); `rm` is the
    rounding mode keyword. Nothing is emitted on error."""
    with asm._forget_names_on_error():
        _encode(asm, mnemonic, ops, rm)


def _encode(asm: Any, mnemonic: str, ops: Sequence[Any], rm: str | None) -> None:
    params = _flatten(ops)
    template = MAP_OP.get(f"{mnemonic}_{len(params)}")
    if template is None:
        counts = MNEMONIC_ARGC.get(mnemonic)
        if counts is None:
            raise EncodeError(f"unknown instruction {mnemonic!r}")
        raise _error(mnemonic, ops, f"expects {' or '.join(map(str, counts))} operands")
    if rm is not None and mnemonic not in RM_MNEMONICS:
        raise _error(mnemonic, ops, "takes no rounding mode")

    best: _Fail | None = None
    for t in template.split("|"):
        alt = _Alt(asm, params, rm)
        try:
            word = _parse_template(t, alt)
        except _Fail as e:
            if best is None or (e.rank, e.pos) > (best.rank, best.pos):
                best = e
            continue
        if alt.li is not None:
            words = li_words((word >> 7) & 31, alt.li)
        elif alt.pcrel:
            tmp = (word >> 15) & 31
            words = [tmp << 7 | 0x17, word]  # auipc tmp, 0
        else:
            words = [word]
        start = asm.pos()
        asm.emit(b"".join(w.to_bytes(4, "little") for w in words))
        if alt.target is not None:
            # The kind ORs its fields into the words at link time.
            asm.add_patch(start, PCREL32 if alt.pcrel else alt.branch, alt.target)
        # objdump leaves out the dynamic rounding mode.
        shown = alt.shown if rm in (None, "dyn") else [*alt.shown, rm]
        asm.note_insn(start, mnemonic, tuple(shown))
        return
    assert best is not None
    raise _error(mnemonic, ops, best.msg)
