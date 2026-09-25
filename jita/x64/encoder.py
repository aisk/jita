"""x64 template interpreter: (mnemonic, operands) -> bytes + patches.

This is a port of the template matching and pattern processing parts of
DynASM's ``dasm_x86.lua`` (``parseoperand``, the ``.template__`` matching
loop, ``dopattern``, ``wputop`` and ``wputmrmsib``), restricted to x64 mode.

DynASM splits the work between a Lua preprocessor (templates -> action list)
and a C runtime (``dasm_x86.h``) that fills in values known only at runtime.
In jita every operand value is known when the instruction is encoded, so the
runtime-only actions collapse into their constant-operand behavior:

- Immediates are always constants, so the ``I`` template character (which
  in DynASM emits an action that picks imm8 vs imm32 at runtime) never
  matches. Constant immediates select the ``S`` (imm8) or ``i`` forms at
  match time, exactly like DynASM does for numeric literals.
- Displacements are always constants, so ModRM mod bits are chosen here
  (DynASM does the same for numeric displacements).
- There are no variable registers (``Rq(expr)``), so all VREG machinery is
  absent.

Deliberate differences from DynASM:

- Branches are never relaxed. DynASM's ``dasm_link`` silently shrinks
  ``jmp``/``jcc`` rel32 to rel8 when the target is in range. jita keeps rel32
  unless the caller asks for the short form (``jmp.short``), which then
  always emits rel8 and fails at link time if out of range.
- Rip-relative operands may be followed by an immediate (DynASM raises NYI).
  The REL32 addend is corrected for the trailing immediate bytes.
- Immediates are range checked against their mathematical value. DynASM
  treats 32 bit literals as bit patterns, so ``add rax, 0xffffffff`` would
  silently become ``add rax, -1``; jita rejects it. ``mov r64, imm`` with an
  immediate outside the int32 range (or a Label/Extern) is encoded as
  ``mov64`` (``movabs``, REX.W B8+r imm64) instead of failing or truncating.
- Byte and word sized ``i`` immediates also accept negative values
  (``mov al, -1``), which DynASM rejects as out of range.
- ``push imm`` only accepts int32 values; DynASM treats 0xffffff80..0xffffffff
  as negative numbers.
- 32 bit address registers (``[eax]``) get an address size prefix (0x67).
  DynASM x64 accepts them but encodes a 64 bit address.
- ``xchg eax, eax`` is encoded as 87 C0, never as the 90 short form, which
  on x64 is a plain nop and does not zero-extend eax into rax.
- Absolute memory operands (``qword[addr]``) may carry a 64 bit address,
  but only ``mov64`` (A0-A3 moffs64) can encode one; ModRM forms reject
  addresses that are not a sign-extended int32 (``0xffffffff80000000`` is
  accepted as ``-0x80000000``).
- Combining ah/ch/dh/bh with any operand that requires a REX prefix is an
  error (DynASM only checks the spl/bpl/sil/dil mix).

Note for phase 2 (template holes): like DynASM, the encoding depends on
register numbers and immediate values, not just operand kinds. rbp/r13
bases force a disp8, rsp/r12 bases force a SIB byte, registers 8..15 and
spl..dil add a REX prefix, extended registers turn a 2 byte VEX prefix into
a 3 byte one, the accumulator and cl select dedicated opcodes (``add rax,
imm32`` is 48 05, ``shl r/m, cl`` needs cl), and immediates in -128..127
select imm8 forms. A hole standing for a register or an immediate therefore
fixes the instruction length only if the hole's range is restricted
accordingly.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import Any

from ..core.errors import EncodeError
from ..core.labels import Extern, Label
from ..core.operand import Hole, Imm
from ..core.patch import ABS64, REL8, REL32, PatchKind
from .mem import MemExpr
from .regs import Reg
from .table import MAP_OP

__all__ = ["encode", "MNEMONIC_ARGC"]

# Operand size codes (DynASM map_opsize / map_opsizenum).
_REG_OPSIZE = {("gp", 1): "b", ("gp", 2): "w", ("gp", 4): "d", ("gp", 8): "q", ("xmm", 16): "o", ("ymm", 32): "y", ("st", 10): "f"}
_MEM_OPSIZE = {1: "b", 2: "w", 4: "d", 8: "q", 10: "t", 16: "o", 32: "y"}
_XSC = {1: 0, 2: 1, 4: 2, 8: 3}

# VEX template characters -> which operand supplies vvvv (1-based), or False.
_VEXARG = {"u": False, "v": 1, "V": 2, "w": 3}

_HEX = frozenset("0123456789abcdefABCDEF")

# Valid ranges (mathematical value) of an immediate for a given operation
# size. None is "no size" (push imm), which the CPU sign-extends to 64 bits.
_INT32 = (-(1 << 31), (1 << 31) - 1)
_SIZE_RANGE = {
    "b": (-(1 << 7), (1 << 8) - 1),
    "w": (-(1 << 15), (1 << 16) - 1),
    "d": (-(1 << 31), (1 << 32) - 1),
    None: _INT32,
    "q": _INT32,
}


_LABEL_IMM_WHY = (
    "a Label/Extern immediate is only valid as a jump/call target "
    "or as the source of mov r64 / mov64"
)

# xchg eax, eax must not use the 90 short form: on x64, 90 is a true nop,
# while 87 C0 zero-extends eax into rax.
_XCHG_NO_SHORT = "rm:87rM|mr:87Rm"


def _build_argc() -> dict[str, tuple[int, ...]]:
    out: dict[str, list[int]] = {}
    for key in MAP_OP:
        name, _, argc = key.rpartition("_")
        out.setdefault(name, []).append(int(argc))
    return {k: tuple(sorted(v)) for k, v in out.items()}


# Mnemonic -> accepted operand counts, e.g. "imul" -> (1, 2, 3).
MNEMONIC_ARGC: dict[str, tuple[int, ...]] = _build_argc()


class _Arg:
    """A classified operand, the equivalent of DynASM's parseoperand() table.

    mode:    DynASM mode string ("rm", "rmR", "rmC", "xm", "i", "iS", "i1S", "iJ")
    opsize:  size code ("b" "w" "d" "q" "o" "y" "t") or None
    reg:     register code, or base register code for memory (None if no base)
    xreg:    index register code (memory only)
    xsc:     index scale as a 2 bit shift count
    disp:    displacement (memory) or rip-relative addend
    riprel:  memory operand is rip-relative
    label:   rip-relative Label target
    imm:     immediate after DynASM's 32 bit wrap normalization
    raw:     immediate as given
    target:  Label/Extern for "iJ"
    needrex: True for spl..dil/r8b.., False for ah..bh, None otherwise
    """

    __slots__ = (
        "op", "mode", "opsize", "reg", "xreg", "xsc", "disp", "riprel",
        "label", "imm", "raw", "target", "needrex", "high",
    )  # fmt: skip

    def __init__(self, op: Any):
        self.op = op
        self.opsize = None
        self.reg = self.xreg = self.xsc = None
        self.disp = 0
        self.riprel = False
        self.label = self.imm = self.raw = self.target = None
        self.needrex = None
        self.high = False


def _classify(op: Any, mnemonic: str, ops: Sequence[Any]) -> _Arg:
    a = _Arg(op)
    if isinstance(op, Reg):
        sz = _REG_OPSIZE.get((op.kind, op.size))
        if sz is None:
            raise _fail(mnemonic, ops, f"{op} cannot be used as an operand")
        a.opsize = sz
        a.reg = op.code
        if sz == "f":  # x87 stack register, "F" marks st0
            a.mode = "fF" if op.code == 0 else "f"
        elif op.code == 0:
            a.mode = "rmR"
        elif op.code == 1 and sz == "b" and not op.high:
            a.mode = "rmC"
        else:
            a.mode = "rm"
        if sz == "b":
            if op.high:
                a.needrex, a.high = False, True
            elif op.code >= 4:
                a.needrex = True
        return a
    if isinstance(op, MemExpr):
        a.mode = "xm"
        if op.size is not None:
            a.opsize = _MEM_OPSIZE.get(op.size)
            if a.opsize is None:
                raise _fail(mnemonic, ops, f"bad memory operand size {op.size}")
        base, index = op.base, op.index
        if base is not None and base.kind == "rip":
            a.riprel = True
            a.label = op.label
        elif base is not None:
            a.reg = base.code
        if index is not None:
            a.xreg = index.code
            a.xsc = _XSC[op.scale]
        a.disp = op.disp
        return a
    if isinstance(op, (Label, Extern)):
        a.mode = "iJ"
        a.opsize = "q"
        a.target = op
        return a
    if isinstance(op, Hole):
        raise _fail(mnemonic, ops, "template holes are not supported yet")
    if isinstance(op, Imm) or (isinstance(op, int) and not isinstance(op, bool)):
        raw = Imm.coerce(op).value
        imm = raw
        # DynASM: numbers are 32 bit patterns, 0xffffff80..0xffffffff are imm8.
        if 4294967168 <= imm <= 4294967295:
            imm -= 4294967296
        m = "i"
        if imm == 1:
            m += "1"
        if -128 <= imm <= 127:
            m += "S"
        a.mode, a.imm, a.raw = m, imm, raw
        return a
    raise _fail(mnemonic, ops, f"unsupported operand {op!r}")


def _fail(mnemonic: str, ops: Sequence[Any], why: str | None = None) -> EncodeError:
    msg = f"{mnemonic}: no encoding for ({', '.join(repr(o) for o in ops)})"
    if why:
        msg += f": {why}"
    return EncodeError(msg)


def _matchtm(tm: str, args: list[_Arg]) -> bool:
    for i, a in enumerate(args):
        c = tm[i]
        if c != "." and c not in a.mode:
            return False
    return True


class _Vex:
    __slots__ = ("m", "p", "v", "l")

    def __init__(self, m: int, p: int, v: _Arg | None):
        self.m, self.p, self.v, self.l = m, p, v, False


class _Encoder:
    """Emits one instruction into a private buffer (DynASM's dopattern)."""

    def __init__(self, mnemonic: str, ops: Sequence[Any], args: list[_Arg], short: bool):
        self.mnemonic = mnemonic
        self.ops = ops
        self.args = args
        self.short = short
        self.used_j = False
        self.has_high = any(a.high for a in args)
        self.out = bytearray()
        # (offset, kind, target, addend, riprel)
        self.fixups: list[tuple[int, PatchKind, Any, int, bool]] = []

    def fail(self, why: str) -> EncodeError:
        return _fail(self.mnemonic, self.ops, why)

    # -- low level output ---------------------------------------------------

    def putb(self, n: int) -> None:
        self.out.append(n & 0xFF)

    def putd(self, n: int) -> None:
        self.out += (n & 0xFFFFFFFF).to_bytes(4, "little")

    def fixup(self, kind: PatchKind, target: Any, addend: int = 0, riprel: bool = False) -> None:
        self.fixups.append((len(self.out), kind, target, addend, riprel))
        self.out += bytes(kind.size)

    def put_sbyte(self, a: _Arg) -> None:
        n = a.imm
        if not -128 <= n <= 127:
            raise self.fail("signed immediate byte out of range")
        self.putb(n)

    def put_ubyte(self, a: _Arg) -> None:
        n = a.imm
        if not 0 <= n <= 255:
            raise self.fail("unsigned immediate byte out of range")
        self.putb(n)

    def put_uword(self, a: _Arg) -> None:
        n = a.imm
        if not 0 <= n <= 0xFFFF:
            raise self.fail("unsigned immediate word out of range")
        self.out += n.to_bytes(2, "little")

    def put_szarg(self, sz: str | None, a: _Arg) -> None:
        """DynASM wputszarg for the "i" pattern character."""
        n = a.imm
        if sz is None or sz == "d" or sz == "q":
            self.putd(n)
        elif sz == "w":
            if not -(1 << 15) <= n < (1 << 16):
                raise self.fail("immediate word out of range")
            self.out += (n & 0xFFFF).to_bytes(2, "little")
        elif sz == "b":
            if not -128 <= n <= 255:
                raise self.fail("immediate byte out of range")
            self.putb(n)
        else:
            raise self.fail("bad operand size")

    # -- opcode, REX and VEX (wputop) ---------------------------------------

    def putop(self, sz: str | None, op: int, rex: int, vex: _Vex | None) -> None:
        if vex is not None:
            tail = None
            if vex.m == 1 and rex & 11 == 0:
                self.putb(0xC5)
                tail = ((rex & 4) ^ 4) << 5
            if tail is None:
                self.putb(0xC4)
                self.putb((((rex & 7) ^ 7) << 5) + vex.m)
                tail = (rex & 8) << 4
            reg = 0
            if vex.v is not None:
                if vex.v.mode[0] != "r":
                    raise self.fail("bad vex operand")
                reg = vex.v.reg
            if sz == "y" or vex.l:
                tail += 4
            self.putb(tail + ((reg ^ 15) << 3) + vex.p)
            rex = 0
            if op >= 256:
                raise self.fail("bad vex opcode")
        elif rex != 0 and self.has_high:
            raise self.fail("ah/ch/dh/bh cannot be encoded with a REX prefix")
        if sz == "w":
            self.putb(0x66)
        if op >= 1 << 32:
            self.putb(op >> 32)
            op &= 0xFFFFFFFF
        if op >= 1 << 24:
            self.putb(op >> 24)
            op &= 0xFFFFFF
        if op >= 1 << 16:
            if rex != 0:
                opc3 = op & 0xFFFF00
                if opc3 == 0x0F3A00 or opc3 == 0x0F3800:
                    self.putb(0x40 + (rex & 15))
                    rex = 0
            self.putb(op >> 16)
            op &= 0xFFFF
        if op >= 256:
            b = op >> 8
            if b == 15 and rex != 0:
                self.putb(0x40 + (rex & 15))
                rex = 0
            self.putb(b)
            op &= 255
        if rex != 0:
            self.putb(0x40 + (rex & 15))
        if sz == "b":
            op -= 1
        self.putb(op)

    # -- ModRM / SIB / displacement (wputmrmsib) ----------------------------

    @staticmethod
    def modrm(m: int, s: int, rm: int) -> int:
        return (m << 6) | ((s & 7) << 3) | (rm & 7)

    def putmrmsib(self, t: _Arg, s: int) -> None:
        if t.mode[0] == "r":
            self.putb(self.modrm(3, s, t.reg))
            return
        if t.riprel:
            # [rip+disp] or [rip+label+addend] -> (0, s, ebp)
            self.putb(self.modrm(0, s, 5))
            if t.label is not None:
                self.fixup(REL32, t.label, t.disp, riprel=True)
            else:
                self.putd(t.disp)
            return
        reg, xreg, disp = t.reg, t.xreg, t.disp
        if reg is None:
            self.putb(self.modrm(0, s, 4))
            if xreg is not None:
                # [xreg*xsc+disp] -> (0, s, esp) (xsc, xreg, ebp)
                self.putb(self.modrm(t.xsc, xreg, 5))
            else:
                # [disp] -> (0, s, esp) (0, esp, ebp)
                self.putb(self.modrm(0, 4, 5))
            if disp >= 1 << 63:
                # An address spelled as unsigned 64 bits (0xffffffff80000000)
                # is the same address as its sign-extended disp32 form.
                disp -= 1 << 64
            if not -(1 << 31) <= disp < (1 << 31):
                raise self.fail("absolute address does not fit in 32 bits, use mov64")
            self.putd(disp)
            return
        if disp == 0 and reg & 7 != 5:
            m = 0
        elif -128 <= disp <= 127:
            m = 1
        else:
            m = 2
        if xreg is not None or reg & 7 == 4:
            # Index register present or esp as base register: need SIB.
            self.putb(self.modrm(m, s, 4))
            self.putb(self.modrm(t.xsc or 0, 4 if xreg is None else xreg, reg))
        else:
            self.putb(self.modrm(m, s, reg))
        if m == 1:
            self.putb(disp)
        elif m == 2:
            self.putd(disp)

    # -- pattern processing (dopattern) -------------------------------------

    def dopattern(self, pat: str, sz: str | None, needrex: bool | None) -> None:
        args = list(self.args)
        opcode: int | None = 0
        addin: _Arg | None = None
        vex: _Vex | None = None
        szov = sz
        narg = 1  # 1-based, like the Lua code
        rex = 0

        for c in pat + "|":
            if c in _HEX:
                opcode = (opcode or 0) * 16 + int(c, 16)
                addin = None
            elif c == "n":  # Disable operand size mods for opcode.
                szov = None
            elif c == "X":  # Force REX.W.
                rex = 8
            elif c == "L":  # Force VEX.L.
                assert vex is not None
                vex.l = True
            elif c == "r":  # Merge 1st operand regno. into opcode.
                addin = args[0]
                opcode += addin.reg % 8
                narg = max(narg, 2)
            elif c == "R":  # Merge 2nd operand regno. into opcode.
                addin = args[1]
                opcode += addin.reg % 8
                narg = 3
            elif c == "m" or c == "M":  # Encode ModRM/SIB.
                if addin is not None:
                    s = addin.reg
                    opcode -= s & 7  # Undo regno opcode merge.
                else:
                    s = opcode & 15  # Undo last digit.
                    opcode >>= 4
                nn = 1 if c == "m" else 2
                t = args[nn - 1]
                narg = max(narg, nn + 1)
                if szov == "q" and rex == 0:
                    rex += 8
                if t.reg is not None and t.reg > 7:
                    rex += 1
                if t.xreg is not None and t.xreg > 7:
                    rex += 2
                if s > 7:
                    rex += 4
                if needrex:
                    rex += 16
                self.putop(szov, opcode, rex, vex)
                opcode = None
                self.putmrmsib(t, s)
                addin = None
            elif c in _VEXARG:  # Encode using VEX prefix.
                b = opcode & 255
                opcode >>= 8
                m = 1
                if b == 0x38:
                    m = 2
                elif b == 0x3A:
                    m = 3
                if m != 1:
                    b = opcode & 255
                    opcode >>= 8
                if b != 0x0F:
                    raise self.fail(f"expected 0F, 0F38 or 0F3A before {c!r} in pattern {pat!r}")
                v = _VEXARG[c]
                varg = args.pop(v - 1) if v else None
                b = opcode & 255
                p = {0x66: 1, 0xF3: 2, 0xF2: 3}.get(b, 0)
                if p:
                    opcode >>= 8
                if opcode != 0:
                    self.putop(None, opcode, 0, None)
                    opcode = 0
                vex = _Vex(m, p, varg)
            else:
                if opcode is not None:  # Flush opcode.
                    if szov == "q" and rex == 0:
                        rex += 8
                    if needrex:
                        rex += 16
                    if addin is not None and addin.reg > 7:
                        rex += 1
                    self.putop(szov, opcode, rex, vex)
                    opcode = None
                if c == "|":
                    break
                if c == "o":  # Offset (pure 32 bit displacement).
                    self.putd(args[0].disp)
                    narg = max(narg, 2)
                elif c == "O":
                    self.putd(args[1].disp)
                    narg = 3
                else:
                    # Anything else is an immediate operand.
                    a = args[narg - 1]
                    narg += 1
                    self.immediate(c, a, sz, pat)

    def immediate(self, c: str, a: _Arg, sz: str | None, pat: str) -> None:
        if a.mode == "iJ" and c != "J":
            raise self.fail(_LABEL_IMM_WHY)
        if c in "SUWiI" and a.imm is not None:
            lo, hi = _SIZE_RANGE.get(sz, (None, None))
            if lo is not None and not lo <= a.raw <= hi:
                raise self.fail(f"immediate {a.raw:#x} out of range for operand size")
        if c == "S":
            self.put_sbyte(a)
        elif c == "U":
            self.put_ubyte(a)
        elif c == "W":
            self.put_uword(a)
        elif c == "i" or c == "I":
            self.put_szarg(sz, a)
        elif c == "J":
            self.used_j = True
            if self.short:
                out = self.out
                if out and out[-1] == 0xE9:  # jmp rel32 -> jmp rel8
                    out[-1] = 0xEB
                elif len(out) >= 2 and out[-2] == 0x0F and out[-1] & 0xF0 == 0x80:  # jcc
                    out[-2:] = bytes([out[-1] - 0x10])
                else:
                    raise self.fail("no short form")
                self.fixup(REL8, a.target)
            else:
                self.fixup(REL32, a.target)
        elif c == "s":  # 4 bit register immediate (is4).
            if a.mode[0] != "r":
                raise self.fail("expected a register operand")
            self.putb(a.reg << 4)
        else:
            raise self.fail(f"bad char {c!r} in pattern {pat!r}")

    # -- final emission -----------------------------------------------------

    def flush(self, asm: Any) -> None:
        out = self.out
        total = len(out)
        start = asm.pos()
        pos = 0
        for off, kind, target, addend, riprel in self.fixups:
            if riprel:
                # REL32 is relative to the end of the field; rip is the end of
                # the instruction, which may have an immediate after the disp.
                addend -= total - (off + kind.size)
            if off > pos:
                asm.emit(bytes(out[pos:off]))
            asm.emit_patch(kind, target, addend)
            pos = off + kind.size
        if pos < total:
            asm.emit(bytes(out[pos:]))
        asm.note_insn(start, self.mnemonic + (".short" if self.short else ""), self.ops)


def _needs_addr32(args: list[_Arg]) -> bool:
    for a in args:
        op = a.op
        if isinstance(op, MemExpr):
            for r in (op.base, op.index):
                if r is not None and r.kind == "gp" and r.size == 4:
                    return True
    return False


def _encode_template(mnemonic: str, template: str, ops: Sequence[Any], short: bool) -> _Encoder:
    args = [_classify(op, mnemonic, ops) for op in ops]

    # Zero-operand opcodes have no match part.
    if not args:
        enc = _Encoder(mnemonic, ops, args, short)
        enc.dopattern(template, "d", None)
        return enc

    # Determine common operand size (coerce undefined size) or flag as mixed.
    sz: str | None = None
    szmix = False
    needrex: bool | None = None
    for a in args:
        nsz = a.opsize
        if nsz is not None:
            if sz is not None and sz != nsz:
                szmix = True
            else:
                sz = nsz
        if a.needrex is not None:
            if needrex is None:
                needrex = a.needrex
            elif needrex != a.needrex:
                raise _fail(mnemonic, ops, "bad mix of byte-addressable registers")

    # Try all match:pattern pairs (separated by '|').
    nargs = len(args)
    gotmatch = False
    lastpat = ""
    for tm in template.split("|"):
        if not tm:
            continue
        colon = tm.index(":", nargs)
        szm, pat = tm[nargs:colon], tm[colon + 1 :]
        if pat == "":
            pat = lastpat
        else:
            lastpat = pat
        if not _matchtm(tm, args):
            continue
        prefix = szm[:1]
        chosen: str | None = None
        matched = False
        if prefix == "/":  # Exactly match leading operand sizes.
            if all(args[j - 1].opsize == szm[j] for j in range(1, len(szm))):
                matched, chosen = True, sz
        else:  # Match common operand size.
            szp = sz
            if szm == "":
                szm = "qdwb"  # Default sizes.
            if prefix == "1":
                szp = args[0].opsize
                szmix = False
            elif prefix == "2":
                szp = args[1].opsize
                szmix = False
            if not szmix and (prefix == "." or (szp is not None and szp in szm)):
                matched, chosen = True, szp
        if matched:
            enc = _Encoder(mnemonic, ops, args, short)
            if _needs_addr32(args):
                enc.putb(0x67)
            enc.dopattern(pat, chosen, needrex)
            return enc
        gotmatch = True

    why = "bad operand mode"
    if any(a.mode == "iJ" for a in args):
        why = _LABEL_IMM_WHY
    elif gotmatch:
        if szmix:
            why = "mixed operand size"
        else:
            why = "bad operand size" if sz is not None else "missing operand size"
    raise _fail(mnemonic, ops, why)


def _encode_mov64(ops: Sequence[Any], mnemonic: str = "mov64") -> _Encoder:
    """DynASM's x64-only ``mov64``: 64 bit immediates and absolute addresses.

    mov64 r64, imm64|Label|Extern   REX.W B8+r imm64 (ABS64 patch for labels)
    mov64 al/ax/eax/rax, [abs]      A0/A1 moffs64
    mov64 [abs], al/ax/eax/rax      A2/A3 moffs64
    """
    if len(ops) != 2:
        raise _fail(mnemonic, ops, "expects 2 operands")
    dst, src = ops
    args = [_classify(op, mnemonic, ops) for op in ops]
    enc = _Encoder(mnemonic, ops, args, False)
    if isinstance(dst, MemExpr) or isinstance(src, MemExpr):
        memarg, rarg = (args[0], args[1]) if isinstance(dst, MemExpr) else (args[1], args[0])
        mem = memarg.op
        if rarg.mode != "rmR" or rarg.opsize not in ("b", "w", "d", "q"):
            raise _fail(mnemonic, ops, "bad operand mode")
        if mem.base is not None or mem.index is not None:
            raise _fail(mnemonic, ops, "moffs operand must be an absolute address")
        if mem.size is not None and memarg.opsize != rarg.opsize:
            raise _fail(mnemonic, ops, "mixed operand size")
        sz = rarg.opsize
        opcode = 0xA3 if isinstance(dst, MemExpr) else 0xA1
        enc.putop(sz, opcode, 8 if sz == "q" else 0, None)
        enc.out += (mem.disp & (2**64 - 1)).to_bytes(8, "little")
        return enc
    r = args[0]
    if not isinstance(dst, Reg) or r.mode[0] != "r" or r.opsize != "q":
        raise _fail(mnemonic, ops, "bad operand mode")
    enc.putop(None, 0xB8 + (r.reg & 7), 9 if r.reg > 7 else 8, None)
    s = args[1]
    if s.mode == "iJ":
        enc.fixup(ABS64, s.target)
    elif s.raw is not None:
        if not -(1 << 63) <= s.raw < (1 << 64):
            raise _fail(mnemonic, ops, "immediate does not fit in 64 bits")
        enc.out += (s.raw & (2**64 - 1)).to_bytes(8, "little")
    else:
        raise _fail(mnemonic, ops, "bad operand mode")
    return enc


def _is_mov64_imm(dst: Any, src: Any) -> bool:
    if not (isinstance(dst, Reg) and dst.kind == "gp" and dst.size == 8):
        return False
    if isinstance(src, (Label, Extern)):
        return True
    if isinstance(src, Imm):
        src = src.value
    if isinstance(src, int) and not isinstance(src, bool):
        return not _INT32[0] <= src <= _INT32[1]
    return False


def _is_eax(op: Any) -> bool:
    return isinstance(op, Reg) and op.kind == "gp" and op.size == 4 and op.code == 0


def _resolve_names(asm: Any, op: Any) -> Any:
    """Turn string label references into the assembler's named labels."""
    if isinstance(op, str):
        return asm.named(op)
    if isinstance(op, MemExpr) and isinstance(op.label, str):
        return replace(op, label=asm.named(op.label))
    return op


def encode(asm: Any, mnemonic: str, ops: Sequence[Any], short: bool = False) -> None:
    """Encode one instruction and emit it into `asm`.

    `mnemonic` is the table name ("and", not "and_"). `short` requests the
    rel8 form of a jmp/jcc to a label. Nothing is emitted if encoding fails.
    """
    ops = tuple(_resolve_names(asm, op) for op in ops)
    if mnemonic in ("mov64", "movabs"):
        enc = _encode_mov64(ops, mnemonic)
    elif mnemonic == "mov" and len(ops) == 2 and _is_mov64_imm(ops[0], ops[1]):
        enc = _encode_mov64(ops, mnemonic)
    else:
        template = MAP_OP.get(f"{mnemonic}_{len(ops)}")
        if mnemonic == "xchg" and len(ops) == 2 and all(_is_eax(op) for op in ops):
            template = _XCHG_NO_SHORT
        if template is None:
            counts = MNEMONIC_ARGC.get(mnemonic)
            if counts is None:
                raise EncodeError(f"unknown instruction {mnemonic!r}")
            raise _fail(mnemonic, ops, f"expects {' or '.join(map(str, counts))} operands")
        enc = _encode_template(mnemonic, template, ops, short)
    if short and not enc.used_j:
        raise _fail(mnemonic, ops, "short form needs a label target")
    enc.flush(asm)
