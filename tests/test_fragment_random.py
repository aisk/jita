"""Randomized fragments: a seeded sample of instruction templates, every
combination of holes in each instruction, several instantiations each.

Every instance must decode (objdump) to the same instruction as the direct
encoding of the instruction written with the concrete operands, up to the
differences of the fixed-length form (a REX prefix, `+riz*1` for a forced
SIB byte, a zero disp8).
"""

import itertools
import random
import re
from dataclasses import replace

from oracle import disassemble, requires_oracle

import jita.x64 as x64
from jita import Assembler, EncodeError, Fragment, Hole
from jita.x64 import MemExpr, byte, dword, gp8, gp16, gp32, gp64, oword, ptr, qword, word, xmm, yword, ymm
from jita.x64.regs import Reg
from jita.x64.table import MAP_OP

SEED = 20260926
N_TEMPLATES = 100
TUPLES_PER_TEMPLATE = 2
INSTANCES_PER_MASK = 3

_GP = {1: gp8, 2: gp16, 4: gp32, 8: gp64}
_SIZES = [byte, word, dword, qword, oword, yword, ptr]
# Immediates away from 1 (shift by 1 has its own form) and from the
# boundaries where GNU tools and jita may choose different widths.
_IMMS = [3, 0x5A, 0x1234, 0x12345678]
_HOLE_CTOR = {("gp", 1): Hole.gp8, ("gp", 2): Hole.gp16, ("gp", 4): Hole.gp32, ("gp", 8): Hole.gp64,
              ("xmm", 16): Hole.xmm, ("ymm", 32): Hole.ymm}  # fmt: skip
_IMM_HOLES = {1: Hole.imm8, 2: Hole.imm16, 4: Hole.imm32, 8: Hole.imm64}
# x87, control flow and string/IO instructions have no useful holes here.
_SKIP = re.compile(r"^(f\w*|j\w*|call|ret|loop\w*|int|enter|leave|in|out|ins[bwd]|outs[bwd]|mov64|movabs)$")


def _templates(rng: random.Random) -> list[tuple[str, int, list[set[str]]]]:
    keys = []
    for key, tmpl in sorted(MAP_OP.items()):
        mnemonic, _, n = key.rpartition("_")
        n = int(n)
        if not 1 <= n <= 3 or _SKIP.match(mnemonic):
            continue
        modes = [set() for _ in range(n)]
        for alt in tmpl.split("|"):
            for j in range(min(n, alt.find(":") if ":" in alt else 0)):
                modes[j].add(alt[j])
        if all(modes):
            keys.append((mnemonic, n, modes))
    return rng.sample(keys, N_TEMPLATES)


def _operand(rng: random.Random, modes: set[str]):
    kinds = []
    if modes & set("rmR"):
        kinds += ["gp", "gp", "xmm", "ymm"]
    if modes & set("mx"):
        kinds += ["mem", "mem"]
    if "i" in modes:
        kinds += ["imm", "imm"]
    kind = rng.choice(kinds or ["gp"])
    if kind == "gp":
        return _GP[rng.choice((1, 2, 4, 8))](rng.randrange(16))
    if kind == "xmm":
        return xmm(rng.randrange(16))
    if kind == "ymm":
        return ymm(rng.randrange(16))
    if kind == "imm":
        return rng.choice(_IMMS)
    base = gp64(rng.randrange(16))
    m = MemExpr(base=base, disp=rng.choice((0, 8, -128, 0x1000)))
    if rng.random() < 0.5:
        m = m + gp64(rng.choice([n for n in range(16) if n != 4])) * rng.choice((1, 2, 4, 8))
    return rng.choice(_SIZES)[m]


def _call(asm: Assembler, mnemonic: str, ops) -> None:
    fn = x64.insns.INSNS.get(mnemonic) or x64.insns.INSNS[mnemonic + "_"]
    fn(*ops, asm=asm)


def _encodes(mnemonic: str, ops) -> bool:
    try:
        _call(Assembler(x64), mnemonic, ops)
    except EncodeError:
        return False
    return True


def _slots(ops) -> list[tuple[int, str]]:
    """Positions that can become holes: (operand index, what)."""
    out = []
    for j, op in enumerate(ops):
        if isinstance(op, Reg) and (op.kind, op.size) in _HOLE_CTOR:
            out.append((j, "reg"))
        elif isinstance(op, MemExpr):
            out.append((j, "base"))
            if op.index is not None:
                out.append((j, "index"))
        elif isinstance(op, int):
            out.append((j, "imm"))
    return out


def _with_holes(ops, mask, imm_size):
    """The operands with the masked positions replaced by holes, and the
    holes by position."""
    ops = list(ops)
    holes = {}
    for n, (j, what) in enumerate(mask):
        op = ops[j]
        name = f"h{n}"
        if what == "reg":
            h = _HOLE_CTOR[op.kind, op.size](name)
            ops[j] = h
        elif what == "imm":
            h = _IMM_HOLES[imm_size](name)
            ops[j] = h
        else:
            h = Hole.gp64(name)
            ops[j] = replace(op, **{what: h})
        holes[j, what] = h
    return ops, holes


def _build(mnemonic, ops, mask):
    """A fragment for `ops` with holes at `mask`, trying immediate hole sizes
    from the smallest one that holds the value."""
    value = next((ops[j] for j, what in mask if what == "imm"), None)
    sizes = [None]
    if value is not None:
        sizes = [s for s, ctor in _IMM_HOLES.items() if ctor("x").range[0] <= value <= ctor("x").range[1]]
    for size in sizes:
        fops, holes = _with_holes(ops, mask, size)
        frag = Fragment(x64)
        try:
            _call(frag, mnemonic, fops)
        except EncodeError:
            continue
        return frag, holes
    return None, None


def _instance_values(rng, ops, holes, first):
    """Hole values and the concrete operands they stand for. The first
    instance uses the original operands; the others random registers.
    Immediates keep their value."""
    concrete = list(ops)
    values = {}
    for (j, what), h in holes.items():
        if what in ("base", "index"):
            v = getattr(ops[j], what)
            if not first:
                v = gp64(rng.choice([c for c in range(16) if what == "base" or c != 4]))
            concrete[j] = replace(concrete[j], **{what: v})
        elif what == "reg" and not first:
            ctor = {"xmm": xmm, "ymm": ymm}.get(h.regclass) or _GP[h.size]
            v = concrete[j] = ctor(rng.randrange(16))
        else:
            v = ops[j]
        values[h.name] = v
    return values, concrete


def _canon(line: str) -> str:
    line = re.sub(r"^rex(\.\w+)? ", "", line)
    line = line.replace("+riz*1", "").replace("+0x0]", "]").replace("movabs ", "mov ")
    if line.startswith("xchg "):
        line = "xchg " + ",".join(sorted(line[5:].split(",")))
    return line


@requires_oracle
def test_random_fragments_decode_like_direct_encodings():
    rng = random.Random(SEED)
    got, want = Assembler(x64), Assembler(x64)
    count = 0
    for mnemonic, n, modes in _templates(rng):
        tuples = []
        for _ in range(200):
            ops = tuple(_operand(rng, modes[j]) for j in range(n))
            if _encodes(mnemonic, ops):
                tuples.append(ops)
                if len(tuples) == TUPLES_PER_TEMPLATE:
                    break
        for ops in tuples:
            slots = _slots(ops)
            for r in range(1, len(slots) + 1):
                for mask in itertools.combinations(slots, r):
                    frag, holes = _build(mnemonic, ops, mask)
                    if frag is None:
                        continue
                    for k in range(INSTANCES_PER_MASK):
                        values, concrete = _instance_values(rng, ops, holes, k == 0)
                        try:
                            _call(want, mnemonic, concrete)
                        except EncodeError:
                            continue
                        frag.instantiate(got, **values)
                        count += 1
    assert count > 500
    mine = [_canon(line) for line in disassemble(bytes(got.cur.buf))]
    theirs = [_canon(line) for line in disassemble(bytes(want.cur.buf))]
    assert len(mine) == len(theirs) == count
    diff = [(g, w) for g, w in zip(mine, theirs) if g != w]
    assert not diff, diff[:10]
