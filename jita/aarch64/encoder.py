"""aarch64 template interpreter: (mnemonic, operands) -> instruction word.

A port of the operand parsers and ``parse_template`` from DynASM's
``dasm_arm64.lua``. DynASM parses operand text; jita receives operand
objects and first flattens them into DynASM's operand list: a shifted or
extended register (`x2 << 3`, `w2.uxtw(2)`) becomes two operands, the
register and its modifier, exactly as `x2, lsl #3` is two operands in
DynASM. Then the "|" separated alternatives of the template are tried in
order and the first one that accepts the operands wins.

Every value is known when jita encodes an instruction, so the actions that
DynASM's C runtime (``dasm_arm64.h``) fills in later collapse into their
constant behavior: immediates are checked and placed here, and only
branch targets become patches (see `jita.aarch64.patch`).

Differences from DynASM. Apart from the first two they are rejections of
operands DynASM encodes into something other than what was written:

- Logical (bitmask) immediates accept every encodable value, 64 bit
  patterns and negative numbers included (`and x0, x1, 0xffff0000ffff0000`,
  `and w0, w1, -2`). DynASM accepts these only through its runtime
  IMM13X/IMM13W actions, not as literals.
- adrp to a label computes the 4KB page delta of the two addresses.
- Immediates are range checked by value. DynASM's Lua parser reduces
  numbers to 32 bits first, so `add x0, x1, #0x100000000` becomes
  `add x0, x1, #0`.
- xzr/wzr are rejected where register 31 means sp (the operands marked
  "p" in the templates, Rn and non flag setting Rd of the extended
  register add/sub forms, the braa/brab modifier). DynASM encodes
  `add x0, xzr, #1` as `add x0, sp, #1`. xzr is not a memory base.
- `mov` with an FP register and an immediate is rejected (DynASM emits
  movz to the w register with the same number).
- `cset`, `csetm`, `cinc`, `cinv` and `cneg` reject `al`: the inverted
  condition nv also means always, so the result would be the opposite.
- Bitfield aliases (`sbfx`, `ubfx`, `bfxil`, `sbfiz`, `ubfiz`, `bfi`)
  check lsb and width, which DynASM turns into a different bitfield
  operation when out of range.
- Reserved encodings are rejected: shift amounts of 32 or more with 32 bit
  registers (shifted register operands, lsr/asr/ror/extr and bitfield
  immediates), extend amounts above 4, tbz/tbnz bit numbers above 31 with
  a w register, uxtx/sxtx applied to a w register in a 64 bit add/sub.
- Loads and stores whose writeback base is also a transfer register, and
  ldp/ldpsw loading the same register twice, are rejected (CONSTRAINED
  UNPREDICTABLE in the architecture).
- An instruction with more operands than its template consumes is an error.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from ..core.errors import EncodeError
from ..core.labels import Extern, Label
from ..core.operand import Imm
from ..core.patch import PatchKind
from .mem import Addr, MemExpr
from .patch import REL14, REL19, REL21_ADR, REL21_ADRP, REL26
from .regs import Mod, Reg, RegMod
from .table import MAP_ALIAS, MAP_BTI, MAP_COND, MAP_EXTEND, MAP_OP, MAP_SHIFT

__all__ = ["encode", "MNEMONIC_ARGC", "imm13", "fpimm"]


def _build_argc() -> dict[str, tuple[int | str, ...]]:
    out: dict[str, set] = {}
    for key in (*MAP_OP, *MAP_ALIAS):
        name, _, argc = key.rpartition("_")
        out.setdefault(name, set()).add(argc if argc == "*" else int(argc))
    return {k: tuple(sorted(v, key=str)) for k, v in out.items()}


# Mnemonic -> accepted operand counts ("*" for any), after flattening.
MNEMONIC_ARGC: dict[str, tuple[int | str, ...]] = _build_argc()


class _Fail(Exception):
    """One template alternative rejected the operands. `rank` is 0 for an
    operand of the wrong kind, 1 for a bad value or size and 2 for sp/zr
    misuse; `pos` is the operand index. The failure with the highest
    (rank, pos) is reported, as the most specific one."""

    def __init__(self, msg: str, pos: int, rank: int = 0):
        super().__init__(msg)
        self.msg, self.pos, self.rank = msg, pos, rank


def _is_int(x: Any) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def _fmt(n: int) -> str:
    return str(n) if -256 < n < 256 else hex(n)


# -- immediates ---------------------------------------------------------------


def imm13(n: int, r64: bool) -> int | None:
    """Encode a logical (bitmask) immediate as N:immr:imms << 10, or None.

    `n` is taken as a 64 bit (r64) or 32 bit two's complement pattern.
    """
    if r64:
        if not -(1 << 63) <= n < (1 << 64):
            return None
        v = n & 0xFFFFFFFFFFFFFFFF
    else:
        if not -(1 << 31) <= n < (1 << 32):
            return None
        v = n & 0xFFFFFFFF
        v |= v << 32
    if v == 0 or v == 0xFFFFFFFFFFFFFFFF:
        return None
    e = 2
    while e < 64:
        elt = v & ((1 << e) - 1)
        rep = elt
        for _ in range(64 // e - 1):
            rep = (rep << e) | elt
        if rep == v:
            break
        e *= 2
    mask = (1 << e) - 1
    elt = v & mask
    k = elt.bit_count()
    ones = (1 << k) - 1
    for r in range(e):
        if ((ones >> r) | (ones << (e - r))) & mask == elt:
            break
    else:
        return None
    imms = ((-2 * e) & 0x3F) | (k - 1)
    return ((1 if e == 64 else 0) << 22) | (r << 16) | (imms << 10)


def fpimm(x: float) -> int | None:
    """Encode an 8 bit FP immediate (fmov) at bits 13-20, or None."""
    if not math.isfinite(x):
        return None
    m, e = math.frexp(x)
    s, e2 = 0, (e - 2) & 7
    if m < 0:
        m, s = -m, 0x00100000
    m = m * 32 - 16
    if m == int(m) and 0 <= m <= 15 and ((e2 ^ 4) - 4) + 2 == e:
        return s + (e2 << 17) + (int(m) << 13)
    return None


# -- per alternative state ------------------------------------------------------


class _Alt:
    """Parses one template alternative (DynASM's parse_template)."""

    def __init__(self, asm: Any, params: list[Any], op: int = 0):
        self.asm = asm
        # Extended register add/sub: Rn, and Rd unless flags are set, are
        # sp when 31 even where the template has no "p".
        self.ext_form = op & 0x1F200000 == 0x0B200000
        self.ext_rd_sp = self.ext_form and not op & 0x20000000
        self.params = params
        self.n = 0
        self.rtype: str | None = None  # DynASM's parse_reg_type
        self.sp_ok: set[int] = set()
        self.fixup: tuple[PatchKind, Any] | None = None

    def fail(self, msg: str, rank: int = 0, pos: int | None = None) -> _Fail:
        return _Fail(msg, self.n if pos is None else pos, rank)

    def param(self, i: int | None = None) -> Any:
        i = self.n if i is None else i
        if i >= len(self.params):
            raise self.fail("missing operand", pos=i)
        return self.params[i]

    # registers

    def reg(self, shift: int, i: int | None = None) -> int:
        i = self.n if i is None else i
        q = self.param(i)
        if not isinstance(q, Reg):
            raise self.fail("expected a register", pos=i)
        if q.kind == "sp":
            if i not in self.sp_ok:
                raise self.fail("sp is not allowed in this position", 1, i)
        elif q.code == 31 and q.kind == "gp" and (
            i in self.sp_ok or (shift == 5 and self.ext_form) or (shift == 0 and self.ext_rd_sp)
        ):
            raise self.fail(f"{q} is not allowed here, register 31 means sp in this position", 2, i)
        rt = q.rt
        if not self.rtype:
            self.rtype = rt
        elif self.rtype != rt:
            raise self.fail("register size mismatch", 1, i)
        return q.code << shift

    def base(self, r: Reg) -> int:
        # parse_reg_base: an x register or sp at bits 5-9.
        if self.rtype:
            if self.rtype != "x":
                raise self.fail("register size mismatch")
        self.rtype = None
        return r.code << 5

    # immediates

    def int_param(self) -> int:
        q = self.param()
        if isinstance(q, Imm):
            q = q.value
        if not _is_int(q):
            raise self.fail("expected an immediate")
        return q

    def imm(self, n: int, bits: int, shift: int, scale: int, signed: bool, what: str) -> int:
        m = n >> scale
        if m << scale == n:
            if signed:
                if -(1 << (bits - 1)) <= m < (1 << (bits - 1)):
                    return (m & ((1 << bits) - 1)) << shift
            elif 0 <= m < (1 << bits):
                return m << shift
        if scale:
            what += f", a multiple of {1 << scale}"
        raise self.fail(f"immediate {_fmt(n)} out of range ({what})", 1)

    def narrow_check(self, n: int, what: str) -> None:
        # jita: with 32 bit registers bit 5 of these 6 bit fields is reserved.
        if self.rtype == "w" and n >= 32:
            raise self.fail(f"{what} {n} out of range for a 32 bit register (0..31)", 1)

    # modifiers

    def mod_param(self) -> Mod:
        q = self.param()
        if not isinstance(q, Mod):
            raise self.fail("expected a shift or extend")
        return q

    def shift(self) -> int:
        q = self.mod_param()
        s = MAP_SHIFT.get(q.kind)
        if s is None or q.amount is None:
            raise self.fail(f"expected a shift (lsl, lsr or asr #n), got {q}")
        self.narrow_check(q.amount, "shift amount")
        return self.imm(q.amount, 6, 10, 0, False, "shift amount 0..63") + (s << 22)

    def extend(self) -> int:
        q = self.mod_param()
        if q.kind == "lsl":
            s = 3 if self.rtype == "x" else 2
        else:
            s = MAP_EXTEND.get(q.kind)
            if s is None:
                raise self.fail(f"expected an extend (uxtb..sxtx or lsl), got {q}")
        if q.kind in ("uxtx", "sxtx"):
            first, prev = self.params[0], self.params[self.n - 1]
            if isinstance(prev, Reg) and prev.rt == "w" and isinstance(first, Reg) and first.rt == "x":
                # jita: the 64 bit extends read all of the register.
                raise self.fail(f"{q.kind} extends a 64 bit register, write the x register", 1)
        if q.amount is None:
            return s << 13
        if q.amount > 4:
            raise self.fail(f"extend amount {q.amount} out of range (0..4)", 1)
        return self.imm(q.amount, 3, 10, 0, False, "extend amount") + (s << 13)

    def lslx16(self) -> int:
        q = self.mod_param()
        if q.kind != "lsl" or q.amount is None:
            raise self.fail(f"expected lsl #n, got {q}")
        n = q.amount
        if n & (~0x30 if self.rtype == "x" else ~0x10):
            allowed = "0, 16, 32 or 48" if self.rtype == "x" else "0 or 16"
            raise self.fail(f"bad shift amount {n} (must be {allowed})", 1)
        return n << 17

    # addresses

    def load(self, op: int) -> int:
        """parse_load: single register loads and stores."""
        m = self.param()
        if not isinstance(m, MemExpr):
            raise self.fail("expected a memory operand")
        if self.n + 1 < len(self.params):
            raise self.fail("too many operands", pos=self.n + 1)
        scale = op >> 30
        op += self.base(m.base)
        self.overlap(m)
        if m.mode == "post":
            return op + self.imm(m.disp, 9, 12, 0, True, "signed 9 bit offset") + 0x400
        if m.mode == "pre":
            return op + self.imm(m.disp, 9, 12, 0, True, "signed 9 bit offset") + 0xC00
        if m.index is not None:
            idx = m.index
            op += (idx.code << 16) + 0x00200800
            mod = m.mod
            if mod is None:
                if idx.rt != "x":
                    raise self.fail("bad index register type")
                return op + 0x6000
            if mod.amount is None or mod.amount == 0:
                pass
            elif mod.amount == scale:
                op += 0x1000
            else:
                raise self.fail(f"bad index scale {mod.amount} (must be 0 or {scale})", 1)
            if idx.rt == "x":
                if mod.kind == "lsl":
                    return op + 0x6000
                if mod.kind == "sxtx":
                    return op + 0xE000
            else:
                if mod.kind == "uxtw":
                    return op + 0x4000
                if mod.kind == "sxtw":
                    return op + 0xC000
            raise self.fail(f"bad extend/shift specifier {mod.kind}")
        n = m.disp
        if n == 0:
            return op + 0x01000000
        mm = n >> scale
        if mm << scale == n and 0 <= mm < 0x1000:
            return op + (mm << 10) + 0x01000000  # scaled, unsigned 12 bit offset
        if -256 <= n < 256:
            return op + ((n & 511) << 12)  # unscaled, signed 9 bit offset
        raise self.fail(
            f"offset {_fmt(n)} out of range (unsigned multiple of {1 << scale} "
            f"up to {4095 << scale}, or -256..255)",
            1,
        )

    def overlap(self, m: MemExpr) -> None:
        # jita: writeback into a register that is also transferred is
        # CONSTRAINED UNPREDICTABLE.
        if m.mode == "offset" or m.base.kind == "sp":
            return
        for r in self.params[: self.n]:
            if isinstance(r, Reg) and r.kind == "gp" and r.code == m.base.code:
                raise self.fail(f"writeback to {m.base}, which is also transferred, is unpredictable", 1)

    def load_pair(self, op: int) -> int:
        """parse_load_pair: ldp/stp."""
        m = self.param()
        if not isinstance(m, MemExpr):
            raise self.fail("expected a memory operand")
        if self.n + 1 < len(self.params):
            raise self.fail("too many operands", pos=self.n + 1)
        if m.index is not None:
            raise self.fail("register pair addresses take an immediate offset, not an index", 1)
        scale = 2 + (op >> (31 - ((op >> 26) & 1)))
        t, t2 = self.params[0], self.params[1]
        if op & 0x00400000 and t is t2:
            raise self.fail(f"loading {t} twice is unpredictable", 1)
        self.overlap(m)
        if m.mode == "post":
            op += 0x00800000
        elif m.mode == "pre":
            op += 0x01800000
        else:
            op += 0x01000000
        op += self.base(m.base)
        return op + self.imm(m.disp, 7, 15, scale, True, "signed 7 bit scaled offset")

    # labels

    def label(self, op: int) -> None:
        q = self.param()
        if isinstance(q, str):
            q = self.asm.named(q)
        elif not isinstance(q, (Label, Extern)):
            raise self.fail("expected a label")
        self.fixup = (_branch_kind(op), q)


def _branch_kind(op: int) -> PatchKind:
    if op & 0x7C000000 == 0x14000000:
        return REL26  # B, BL
    if op >> 24 == 0x54 or op & 0x7E000000 == 0x34000000 or op & 0x3B000000 == 0x18000000:
        return REL19  # B.cond, CBZ, CBNZ, LDR* literal
    if op & 0x7E000000 == 0x36000000:
        return REL14  # TBZ, TBNZ
    if op & 0x9F000000 == 0x10000000:
        return REL21_ADR
    if op & 0x9F000000 == 0x90000000:
        return REL21_ADRP
    raise AssertionError(f"unknown branch type {op:08x}")


def _parse_template(t: str, alt: _Alt) -> int:
    op = int(t[:8], 16)
    for p in t[8:]:
        if p == "D":
            op += alt.reg(0)
            alt.n += 1
        elif p == "N":
            op += alt.reg(5)
            alt.n += 1
        elif p == "M":
            op += alt.reg(16)
            alt.n += 1
        elif p == "A":
            op += alt.reg(10)
            alt.n += 1
        elif p == "m":
            op += alt.reg(16, alt.n - 1)
        elif p == "p":
            alt.sp_ok.add(alt.n)
        elif p == "g":
            if alt.rtype == "x":
                op += 0x80000000
            elif alt.rtype != "w":
                raise alt.fail("bad register type")
            alt.rtype = None
        elif p == "f":
            if alt.rtype == "d":
                op += 0x00400000
            elif alt.rtype != "s":
                raise alt.fail("bad register type")
            alt.rtype = None
        elif p in "xwdsq":
            if alt.rtype != p:
                raise alt.fail("register size mismatch")
            alt.rtype = None
        elif p == "L":
            op = alt.load(op)
            alt.n = len(alt.params)
        elif p == "P":
            op = alt.load_pair(op)
            alt.n = len(alt.params)
        elif p == "B":
            alt.label(op)
            alt.n += 1
        elif p == "I":
            n = alt.int_param()
            if 0 <= n < 0x1000:
                op += n << 10
            elif 0 <= n <= 0xFFF000 and n & 0xFFF == 0:
                op += (n >> 2) + 0x00400000
            else:
                raise alt.fail(f"immediate {_fmt(n)} out of range (12 bit unsigned, optionally shifted left by 12)", 1)
            alt.n += 1
        elif p == "i":
            n = alt.int_param()
            enc = imm13(n, alt.rtype == "x")
            if enc is None:
                raise alt.fail(f"immediate {_fmt(n)} is not a valid logical (bitmask) immediate", 1)
            op += enc
            alt.n += 1
        elif p == "W":
            op += alt.imm(alt.int_param(), 16, 5, 0, False, "16 bit unsigned")
            alt.n += 1
        elif p == "T":
            n = alt.int_param()
            if not 0 <= n <= 63:
                raise alt.fail(f"bit number {n} out of range (0..63)", 1)
            alt.narrow_check(n, "bit number")
            op += ((n & 0x1F) << 19) + (0x80000000 if n >= 32 else 0)
            alt.n += 1
        elif p == "1":
            n = alt.int_param()
            alt.narrow_check(n, "immediate")
            op += alt.imm(n, 6, 16, 0, False, "6 bit unsigned")
            alt.n += 1
        elif p == "2":
            n = alt.int_param()
            alt.narrow_check(n, "immediate")
            op += alt.imm(n, 6, 10, 0, False, "6 bit unsigned")
            alt.n += 1
        elif p == "5":
            op += alt.imm(alt.int_param(), 5, 16, 0, False, "5 bit unsigned")
            alt.n += 1
        elif p == "V":
            op += alt.imm(alt.int_param(), 4, 0, 0, False, "4 bit nzcv flags")
            alt.n += 1
        elif p == "F":
            q = alt.param()
            if not isinstance(q, (int, float)) or isinstance(q, bool):
                raise alt.fail("expected a floating point immediate")
            enc = fpimm(float(q))
            if enc is None:
                raise alt.fail(f"{q!r} is not a valid 8 bit floating point immediate", 1)
            op += enc
            alt.n += 1
        elif p == "Z":
            q = alt.param()
            if not isinstance(q, (int, float)) or isinstance(q, bool):
                raise alt.fail("expected the constant 0")
            if q != 0 or math.copysign(1, q) < 0:
                raise alt.fail("expected the constant 0", 1)
            alt.n += 1
        elif p == "S":
            op += alt.shift()
            alt.n += 1
        elif p == "X":
            op += alt.extend()
            alt.n += 1
        elif p == "R":
            op += alt.lslx16()
            alt.n += 1
        elif p == "C" or p == "c":
            q = alt.param()
            c = MAP_COND.get(q) if isinstance(q, str) else None
            if c is None:
                raise alt.fail("expected a condition (eq, ne, cs/hs, cc/lo, mi, pl, vs, vc, hi, ls, ge, lt, gt, le, al)")
            if p == "c" and c == 14:
                # jita: the inverted condition would be nv, which also means
                # always, so cset/cinc/... with al would do the opposite.
                raise alt.fail("condition al is not allowed here (it is inverted to nv)", 1)
            op += (c ^ (1 if p == "c" else 0)) << 12
            alt.n += 1
        elif p == "t":
            q = alt.param()
            v = MAP_BTI.get(q) if isinstance(q, str) else None
            if v is None:
                raise alt.fail('expected a bti target ("c", "j" or "jc")')
            op += v
            alt.n += 1
        else:
            raise AssertionError(f"bad template character {p!r} in {t!r}")
    if alt.n < len(alt.params):
        raise alt.fail("too many operands")
    if alt.rtype not in (None, "x", "w"):
        # jita: "52800000DW" (mov reg, #imm16) has no type check, so DynASM
        # encodes mov d0, #1 as movz w0, #1.
        raise alt.fail("bad register type", 2, 0)
    return op


# -- aliases -------------------------------------------------------------------


def _alias(kind: str, params: list[Any], fail) -> list[Any]:
    p = list(params)
    if not all(_is_int(x) for x in p[2:]):
        raise fail("expected an immediate")
    r = p[0]
    if not isinstance(r, Reg):
        raise fail("expected a register")
    size = 32 if r.rt == "w" else 64
    if kind != "lsl":
        # jita: DynASM does not check the field, so an out of range lsb or
        # width silently becomes a different bitfield operation.
        lsb, width = p[2], p[3]
        if not (0 <= lsb < size and 1 <= width <= size - lsb):
            raise fail(f"bitfield lsb {lsb}, width {width} out of range for a {size} bit register")
    if kind == "bfx":
        p[3] = p[2] + p[3] - 1
        return p
    if kind == "bfiz":
        p[2], p[3] = (size - p[2]) % size, p[3] - 1
    else:  # lsl #n
        sh = p[2]
        if not 0 <= sh < size:
            raise fail(f"shift amount {sh} out of range (0..{size - 1})")
        p[2:3] = [(size - sh) % size, size - 1 - sh]
    return p


# -- entry point ---------------------------------------------------------------


def _flatten(ops: Sequence[Any]) -> list[Any]:
    out: list[Any] = []
    for op in ops:
        if isinstance(op, RegMod):
            out += (op.reg, op.mod)
        elif isinstance(op, Imm):
            out.append(op.value)
        else:
            out.append(op)
    return out


def _error(mnemonic: str, ops: Sequence[Any], why: str | None = None) -> EncodeError:
    msg = f"{mnemonic}: no encoding for ({', '.join(repr(o) for o in ops)})"
    if why:
        msg += f": {why}"
    return EncodeError(msg)


def encode(asm: Any, mnemonic: str, ops: Sequence[Any], name: str | None = None) -> None:
    """Encode one instruction and emit it into `asm`.

    `mnemonic` is the table name ("and", "beq"). `name` is the name shown
    in listings (defaults to the mnemonic). Nothing is emitted on error.
    """
    for op in ops:
        if isinstance(op, Addr):
            raise _error(mnemonic, ops, f"wrap the address in mem[...]: mem[{op}]")
    params = _flatten(ops)
    key = f"{mnemonic}_{len(params)}"
    alias = MAP_ALIAS.get(key)
    if alias is not None and (key != "lsl_3" or _is_int(params[2])):

        def fail(why):
            return _error(mnemonic, ops, why)

        key = alias[0]
        params = _alias(alias[1], params, fail)
    template = MAP_OP.get(key)
    if template is None:
        template = MAP_OP.get(f"{mnemonic}_*")
    if template is None:
        counts = MNEMONIC_ARGC.get(mnemonic)
        if counts is None:
            raise EncodeError(f"unknown instruction {mnemonic!r}")
        raise _error(mnemonic, ops, f"expects {' or '.join(map(str, counts))} operands")

    best: _Fail | None = None
    for t in template.split("|"):
        alt = _Alt(asm, params, int(t[:8], 16))
        try:
            word = _parse_template(t, alt)
        except _Fail as e:
            if best is None or (e.rank, e.pos) > (best.rank, best.pos):
                best = e
            continue
        if word & 0xFFFFF800 == 0xD71F0800 and word & 31 == 31:
            # jita: the braa/brab modifier register 31 is sp, not xzr.
            raise _error(mnemonic, ops, "the modifier register cannot be xzr (register 31 means sp here)")
        start = asm.pos()
        data = word.to_bytes(4, "little")
        if alt.fixup is None:
            asm.emit(data)
        else:
            # The patch reserves the word; the kind ORs its field into it.
            kind, target = alt.fixup
            asm.emit_patch(kind, target)
            asm.cur.buf[start : start + 4] = data
        asm.note_insn(start, name or mnemonic, tuple(ops))
        return
    assert best is not None
    raise _error(mnemonic, ops, best.msg)
