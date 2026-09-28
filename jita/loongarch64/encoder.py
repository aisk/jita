"""loongarch64 template interpreter: (mnemonic, operands) -> instruction words.

Works like `jita.riscv64.encoder`: the "|" separated alternatives of the
template (`jita.loongarch64.table`) are tried in order and the first one
that accepts the operands wins. Every value is known when an instruction
is encoded, so immediates are checked and placed here, and only label
targets become patches (see `jita.loongarch64.patch`).

Every instruction is one 32 bit word, except the pseudo instructions that
expand to several: `li.w` and `li.d` (1 to 4 words), and the pairs of
`la.local`, `la.pcrel`, `call36`, `tail36` and loads and stores to a
label. A pseudo instruction is one entry in the listing, with all of its
words.

Immediates are range checked by value and never truncated. The listing
shows operands as objdump does: registers with a `$`, signed immediates
in decimal, unsigned ones in hex.
"""

from collections.abc import Sequence
from typing import Any

from ..core.errors import EncodeError
from ..core.labels import Extern, Label
from ..core.operand import Imm
from .patch import B16, B21, B26, CALL36, PCALA, PCREL20
from .regs import F, Fcc, Fcsr, R, Reg
from .table import MAP_OP

__all__ = ["encode", "li_words", "MNEMONIC_ARGC"]


def _build_argc() -> dict[str, tuple[int, ...]]:
    out: dict[str, set[int]] = {}
    for key in MAP_OP:
        name, _, argc = key.rpartition("_")
        out.setdefault(name, set()).add(int(argc))
    return {k: tuple(sorted(v)) for k, v in out.items()}


# Mnemonic -> accepted operand counts.
MNEMONIC_ARGC: dict[str, tuple[int, ...]] = _build_argc()

_REG_WHAT = {
    R: "an integer register",
    F: "a floating point register",
    Fcc: "a condition flag register ($fcc0..$fcc7)",
    Fcsr: "an fcsr register ($fcsr0..$fcsr3)",
}


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
    """An immediate that listings show in hex, as objdump does for
    unsigned fields."""

    __slots__ = ()

    def __str__(self) -> str:
        return hex(self.value)


# -- li.w and li.d -----------------------------------------------------------


def _sext(v: int, bits: int) -> int:
    return v - (1 << bits) if (v >> (bits - 1)) & 1 else v


def li_words(rd: int, value: int, width: int = 64) -> list[int]:
    """The words of `li.d rd, value` (`width` 64) or `li.w rd, value`
    (`width` 32): GNU as' expansion, which splits the value into the
    fields of lu12i.w/ori (or addi.w), lu32i.d and lu52i.d and leaves out
    each part that the sign extension of the parts below already
    produces. `value` is a signed or unsigned bit pattern of `width`
    bits."""
    lo32 = value & 0xFFFFFFFF
    hi32 = (value >> 32) & 0xFFFFFFFF
    if width == 32:
        hi32 = 0xFFFFFFFF if lo32 & 0x80000000 else 0
    # The four parts, low to high: bits 0-11, 12-31, 32-51, 52-63.
    parts = [lo32 & 0xFFF, lo32 >> 12, hi32 & 0xFFFFF, hi32 >> 20]
    ones = [0xFFF, 0xFFFFF, 0xFFFFF, 0xFFF]
    top = [0x800, 0x80000, 0x80000, 0x800]
    zero = [p == 0 for p in parts]
    # Part k is redundant when it is the sign extension of part k - 1.
    sext = [
        (zero[k] != (parts[k] == ones[k])) and (zero[k] != (k > 0 and bool(parts[k - 1] & top[k - 1])))
        for k in range(4)
    ]
    lu12i = 0x14000000 | (parts[1] & 0xFFFFF) << 5 | rd
    ori = 0x03800000 | parts[0] << 10 | rd << 5 | rd
    lu32i = 0x16000000 | parts[2] << 5 | rd
    lu52i = 0x03000000 | parts[3] << 10 | rd << 5 | rd
    if sext[0] and sext[1] and sext[2] and not sext[3]:
        return [0x03000000 | parts[3] << 10 | rd]  # lu52i.d rd, zero, hi12
    out: list[int] = []
    if zero[1] and not zero[0]:
        out.append(0x03800000 | parts[0] << 10 | rd)  # ori rd, zero, lo12
    elif sext[1] and sext[0]:
        out.append(0x00150000 | rd)  # or rd, zero, zero
    elif sext[1]:
        out.append(0x02800000 | parts[0] << 10 | rd)  # addi.w rd, zero, lo12
    elif sext[0]:
        out.append(lu12i)
    else:
        out += [lu12i, ori]
    if not sext[2]:
        out.append(lu32i)
    if not sext[3]:
        out.append(lu52i)
    return out


# -- per alternative state ------------------------------------------------------


class _Alt:
    """Parses one template alternative."""

    def __init__(self, asm: Any, params: list[Any]):
        self.asm = asm
        self.params = params
        self.shown: list[Any] = list(params)  # operands as listings print them
        self.n = 0
        self.target: Label | Extern | None = None
        self.kind: Any = None  # the patch kind of a label operand
        self.pair = ""  # "L", "l" or "O" for a two instruction sequence
        self.li: tuple[int, int] | None = None  # (value, width)

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
            raise self.fail(f"expected {_REG_WHAT[cls]}")
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

    def signed(self, bits: int) -> int:
        lim = 1 << (bits - 1)
        return self.imm(-lim, lim - 1, f"signed {bits} bit, {_fmt(-lim)}..{_fmt(lim - 1)}") & ((1 << bits) - 1)

    def unsigned(self, bits: int) -> int:
        return self.imm(0, (1 << bits) - 1, f"unsigned {bits} bit, 0..{_fmt((1 << bits) - 1)}", True)

    def offset(self, bits: int) -> int:
        """A byte offset, a multiple of 4 whose word count fits `bits`
        signed bits; returns the word count field."""
        lim = 1 << (bits + 1)
        n = self.int_param()
        if n & 3:
            raise self.fail(f"offset {_fmt(n)} is not a multiple of 4", 1)
        if not -lim <= n < lim:
            raise self.fail(f"offset {_fmt(n)} out of range ({_fmt(-lim)}..{_fmt(lim - 4)})", 1)
        self.shown[self.n] = Imm(n)
        self.next()
        return (n >> 2) & ((1 << bits) - 1)

    def bitfield(self, bits: int) -> int:
        """msb and lsb of bstrins and bstrpick."""
        msb = self.unsigned(bits)
        lsb = self.unsigned(bits)
        if msb < lsb:
            raise self.fail(f"msb {msb} is below lsb {lsb}", 1, self.n - 1)
        return msb << 16 | lsb << 10

    # labels

    def label(self, kind: Any) -> None:
        q = self.param()
        if isinstance(q, str):
            q = self.asm.named(q)
        elif not isinstance(q, (Label, Extern)):
            raise self.fail("expected a label")
        self.target = q
        self.kind = kind
        self.next()


def _parse_template(t: str, alt: _Alt) -> int:
    op = int(t[:8], 16)
    for p in t[8:]:
        if p == "D":
            op |= alt.reg(R, 0)
        elif p == "J":
            op |= alt.reg(R, 5)
        elif p == "K":
            op |= alt.reg(R, 10)
        elif p == "d":
            op |= alt.reg(F, 0)
        elif p == "j":
            op |= alt.reg(F, 5)
        elif p == "k":
            op |= alt.reg(F, 10)
        elif p == "a":
            op |= alt.reg(F, 15)
        elif p == "C":
            op |= alt.reg(Fcc, 0)
        elif p == "c":
            op |= alt.reg(Fcc, 5)
        elif p == "E":
            op |= alt.reg(Fcc, 15)
        elif p == "S":
            op |= alt.reg(Fcsr, 0)
        elif p == "s":
            op |= alt.reg(Fcsr, 5)
        elif p == "=":
            op |= (op & 31) << 5
        elif p == "i":
            op |= alt.signed(12) << 10
        elif p == "u":
            op |= alt.unsigned(12) << 10
        elif p == "h":
            op |= alt.signed(16) << 10
        elif p == "z":
            op |= alt.signed(20) << 5
        elif p == "w":
            op |= alt.unsigned(5) << 10
        elif p == "x":
            op |= alt.unsigned(6) << 10
        elif p == "M":
            op |= alt.bitfield(5)
        elif p == "N":
            op |= alt.bitfield(6)
        elif p == "p":
            op |= alt.offset(14) << 10
        elif p == "o":
            op |= alt.offset(16) << 10
        elif p == "q":
            op |= (alt.imm(1, 4, "shift 1..4", True) - 1) << 15
        elif p == "y":
            op |= alt.unsigned(2) << 15
        elif p == "Y":
            op |= alt.unsigned(3) << 15
        elif p == "U":
            op |= alt.unsigned(15)
        elif p == "P":
            op |= alt.unsigned(5)
        elif p == "X":
            n = alt.unsigned(5)
            op |= n << 5 | n
        elif p == "B":
            alt.label(B16)
        elif p == "R":
            alt.label(B21)
        elif p == "T":
            alt.label(B26)
        elif p == "V":
            alt.label(PCREL20)
        elif p in "LlO":
            alt.label(CALL36 if p == "O" else PCALA)
            alt.pair = p
        elif p == "#" or p == "%":
            n = alt.int_param()
            bits = 64 if p == "#" else 32
            if not -(1 << (bits - 1)) <= n < (1 << bits):
                raise alt.fail(f"immediate {_fmt(n)} out of range ({bits} bit)", 1)
            alt.li = (n, bits)
            alt.next()
        elif p == "~":
            if (op >> 5) & 31 == op & 31:
                raise alt.fail("the pcalau12i register must differ from the register stored", 1, len(alt.params) - 1)
        elif p == "!":
            rd, rk, rj = op & 31, (op >> 10) & 31, (op >> 5) & 31
            if rd and rd in (rk, rj):
                raise alt.fail("rd must differ from rk and rj (unless it is zero)", 1, 0)
        else:
            raise AssertionError(f"bad template character {p!r} in {t!r}")
    if alt.n < len(alt.params):
        raise alt.fail("too many operands")
    if alt.pair in ("L", "O") and not (op >> 5) & 31:
        # The address would be computed into zero and lost: the second
        # instruction would use address 0 plus the low bits.
        raise alt.fail("zero cannot hold the address of a label reference", 1, len(alt.params) - 1)
    return op


# -- entry point ---------------------------------------------------------------


def _flatten(ops: Sequence[Any]) -> list[Any]:
    return [op.value if isinstance(op, Imm) else op for op in ops]


def _error(mnemonic: str, ops: Sequence[Any], why: str | None = None) -> EncodeError:
    msg = f"{mnemonic}: no encoding for ({', '.join(repr(o) for o in ops)})"
    if why:
        msg += f": {why}"
    return EncodeError(msg)


def encode(asm: Any, mnemonic: str, ops: Sequence[Any]) -> None:
    """Encode one instruction (or pseudo instruction) and emit it into
    `asm`. `mnemonic` is the table name ("add.d", "and"). Nothing is
    emitted on error."""
    with asm._forget_names_on_error():
        _encode(asm, mnemonic, ops)


def _encode(asm: Any, mnemonic: str, ops: Sequence[Any]) -> None:
    params = _flatten(ops)
    template = MAP_OP.get(f"{mnemonic}_{len(params)}")
    if template is None:
        counts = MNEMONIC_ARGC.get(mnemonic)
        if counts is None:
            raise EncodeError(f"unknown instruction {mnemonic!r}")
        raise _error(mnemonic, ops, f"expects {' or '.join(map(str, counts))} operands")

    best: _Fail | None = None
    for t in template.split("|"):
        alt = _Alt(asm, params)
        try:
            word = _parse_template(t, alt)
        except _Fail as e:
            if best is None or (e.rank, e.pos) > (best.rank, best.pos):
                best = e
            continue
        if alt.li is not None:
            words = li_words(word & 31, *alt.li)
        elif alt.pair:
            tmp = (word >> 5) & 31
            # pcalau12i or pcaddu18i tmp, 0
            words = [(0x1E000000 if alt.pair == "O" else 0x1A000000) | tmp, word]
        else:
            words = [word]
        start = asm.pos()
        asm.emit(b"".join(w.to_bytes(4, "little") for w in words))
        if alt.target is not None:
            # The kind ORs its fields into the words at link time.
            asm.add_patch(start, alt.kind, alt.target)
        asm.note_insn(start, mnemonic, tuple(alt.shown))
        return
    assert best is not None
    raise _error(mnemonic, ops, best.msg)
