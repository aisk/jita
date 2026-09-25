"""aarch64 encoder tests.

`CASES` and `LABEL_CASES` hold byte-exact expectations produced by running
LuaJIT's DynASM (dynasm.lua with dasm_arm64.lua, and dasm_arm64.h) on the
same instructions, so they pin jita to DynASM's choice among equivalent
encodings. The oracle test then checks every case against an aarch64
assembler (GNU as, or llvm-mc when the cross binutils are missing):
either the bytes are identical, or the case is a listed divergence where
DynASM picks a different but equivalent encoding or accepts a spelling
the GNU syntax does not.
"""

import pytest
from oracle import OracleError, aarch64_assembler, assemble, requires_aarch64_oracle

import jita.aarch64 as A
from jita import Assembler, EncodeError, Extern, Label, LinkError
from jita.aarch64 import *  # noqa: F403
from jita.aarch64.patch import REL14, REL19, REL21_ADR, REL21_ADRP, REL26
from jita.aarch64.insns import INSNS
from jita.aarch64.table import MAP_COND, MAP_OP

LBL = "LBL"  # replaced by a Label in LABEL_CASES


def encode(fn, *ops):
    """Encode one instruction into a fresh Assembler, return (bytes, patches)."""
    a = Assembler(A)
    fn(*ops, asm=a)
    return bytes(a.cur.buf), a.cur.patches


def hexof(fn, *ops):
    return encode(fn, *ops)[0].hex()


# fmt: off
CASES = [
    (add, (x0, x1, x2), "2000028b"),  # add x0, x1, x2
    (add, (w3, w4, w5), "8300050b"),  # add w3, w4, w5
    (add, (x0, x1, 0), "20000091"),  # add x0, x1, #0
    (add, (x0, x1, 4095), "20fc3f91"),  # add x0, x1, #4095
    (add, (x0, x1, 4096), "20044091"),  # add x0, x1, #4096
    (add, (w0, w1, 0xfff000), "20fc7f11"),  # add w0, w1, #16773120
    (add, (sp, sp, 16), "ff430091"),  # add sp, sp, #16
    (add, (x29, sp, 0), "fd030091"),  # add x29, sp, #0
    (add, (x0, x1, x2 << 3), "200c028b"),  # add x0, x1, x2, lsl #3
    (add, (w0, w1, w2 >> 31), "207c420b"),  # add w0, w1, w2, lsr #31
    (add, (x0, x1, x2.asr(63)), "20fc828b"),  # add x0, x1, x2, asr #63
    (add, (x0, x1, w2.uxtw()), "2040228b"),  # add x0, x1, w2, uxtw
    (add, (x0, x1, w2.sxtw(2)), "20c8228b"),  # add x0, x1, w2, sxtw #2
    (add, (x0, sp, x1), "e063218b"),  # add x0, sp, x1
    (add, (x0, sp, x1 << 2), "e06b218b"),  # add x0, sp, x1, lsl #2
    (add, (sp, x1, x2.uxtx(4)), "3f70228b"),  # add sp, x1, x2, uxtx #4
    (add, (w0, w1, w2.uxtb(1)), "2004220b"),  # add w0, w1, w2, uxtb #1
    (add, (x0, x1, x2.sxtx()), "20e0228b"),  # add x0, x1, x2, sxtx
    (adds, (x0, x1, 1), "200400b1"),  # adds x0, x1, #1
    (adds, (xzr, sp, 1), "ff0700b1"),  # adds xzr, sp, #1
    (adds, (w0, w1, w2.lsl(4)), "2010022b"),  # adds w0, w1, w2, lsl #4
    (adds, (x0, sp, x1), "e06321ab"),  # adds x0, sp, x1
    (cmn, (x0, 1), "1f0400b1"),  # cmn x0, #1
    (cmn, (sp, x1), "ff6321ab"),  # cmn sp, x1
    (cmn, (w0, w1 << 2), "1f08012b"),  # cmn w0, w1, lsl #2
    (cmn, (x0, w1.sxth(1)), "1fa421ab"),  # cmn x0, w1, sxth #1
    (sub, (x0, x1, x2), "200002cb"),  # sub x0, x1, x2
    (sub, (sp, sp, 0x10000), "ff4340d1"),  # sub sp, sp, #65536
    (sub, (x0, x1, x2.lsr(7)), "201c42cb"),  # sub x0, x1, x2, lsr #7
    (sub, (x0, sp, w1.uxtw(3)), "e04f21cb"),  # sub x0, sp, w1, uxtw #3
    (subs, (x0, x1, 4095), "20fc3ff1"),  # subs x0, x1, #4095
    (subs, (w0, w1, w2), "2000026b"),  # subs w0, w1, w2
    (cmp, (x0, x1), "1f0001eb"),  # cmp x0, x1
    (cmp, (w0, 0), "1f000071"),  # cmp w0, #0
    (cmp, (sp, 8), "ff2300f1"),  # cmp sp, #8
    (cmp, (x0, x1.asr(5)), "1f1481eb"),  # cmp x0, x1, asr #5
    (cmp, (x0, w1.sxtw()), "1fc021eb"),  # cmp x0, w1, sxtw
    (neg, (x0, x1), "e00301cb"),  # neg x0, x1
    (neg, (w0, w1 << 2), "e00b014b"),  # neg w0, w1, lsl #2
    (negs, (x0, x1), "e00301eb"),  # negs x0, x1
    (negs, (x0, x1.asr(3)), "e00f81eb"),  # negs x0, x1, asr #3
    (adc, (x0, x1, x2), "2000029a"),  # adc x0, x1, x2
    (adcs, (w0, w1, w2), "2000023a"),  # adcs w0, w1, w2
    (sbc, (x0, x1, x2), "200002da"),  # sbc x0, x1, x2
    (sbcs, (w0, w1, w2), "2000027a"),  # sbcs w0, w1, w2
    (ngc, (x0, x1), "e00301da"),  # ngc x0, x1
    (ngcs, (w0, w1), "e003017a"),  # ngcs w0, w1
    (and_, (x0, x1, x2), "2000028a"),  # and x0, x1, x2
    (and_, (x0, x1, 0xff), "201c4092"),  # and x0, x1, #255
    (and_, (w0, w1, 0xfffffffe), "20781f12"),  # and w0, w1, #4294967294
    (and_, (x0, x1, 0xffffffff), "207c4092"),  # and x0, x1, #4294967295
    (and_, (sp, x1, 0xfffffffffffffff0), "3fec7c92"),  # and sp, x1, #18446744073709551600
    (and_, (x0, x1, 0x5555555555555555), "20f00092"),  # and x0, x1, #6148914691236517205
    (and_, (x0, x1, x2 << 1), "2004028a"),  # and x0, x1, x2, lsl #1
    (orr, (x0, x1, x2), "200002aa"),  # orr x0, x1, x2
    (orr, (w0, w1, 0x80000000), "20000132"),  # orr w0, w1, #2147483648
    (orr, (x0, xzr, 0x0000ffff0000ffff), "e03f00b2"),  # orr x0, xzr, #281470681808895
    (orr, (x0, x1, x2.asr(2)), "200882aa"),  # orr x0, x1, x2, asr #2
    (eor, (x0, x1, x2), "200002ca"),  # eor x0, x1, x2
    (eor, (w0, w1, 1), "20000052"),  # eor w0, w1, #1
    (eor, (x0, x1, x2 >> 4), "201042ca"),  # eor x0, x1, x2, lsr #4
    (ands, (x0, x1, 0xf0), "200c7cf2"),  # ands x0, x1, #240
    (ands, (w0, w1, w2), "2000026a"),  # ands w0, w1, w2
    (ands, (x0, x1, x2 << 60), "20f002ea"),  # ands x0, x1, x2, lsl #60
    (tst, (x0, x1), "1f0001ea"),  # tst x0, x1
    (tst, (w0, 4), "1f001e72"),  # tst w0, #4
    (tst, (x0, x1 << 3), "1f0c01ea"),  # tst x0, x1, lsl #3
    (bic, (x0, x1, x2), "2000228a"),  # bic x0, x1, x2
    (bic, (w0, w1, w2 >> 1), "2004620a"),  # bic w0, w1, w2, lsr #1
    (orn, (x0, x1, x2), "200022aa"),  # orn x0, x1, x2
    (orn, (w0, w1, w2.asr(3)), "200ca22a"),  # orn w0, w1, w2, asr #3
    (eon, (x0, x1, x2), "200022ca"),  # eon x0, x1, x2
    (eon, (x0, x1, x2 << 9), "202422ca"),  # eon x0, x1, x2, lsl #9
    (bics, (x0, x1, x2), "200022ea"),  # bics x0, x1, x2
    (bics, (w0, w1, w2 << 1), "2004226a"),  # bics w0, w1, w2, lsl #1
    (movn, (x0, 0), "00008092"),  # movn x0, #0
    (movn, (w0, 0xffff, Mod("lsl", 16)), "e0ffbf12"),  # movn w0, #65535, lsl #16
    (movz, (x0, 0x1234), "804682d2"),  # movz x0, #4660
    (movz, (x0, 0x1234, Mod("lsl", 48)), "8046e2d2"),  # movz x0, #4660, lsl #48
    (movz, (w1, 1, Mod("lsl", 16)), "2100a052"),  # movz w1, #1, lsl #16
    (movk, (x0, 0xbeef, Mod("lsl", 32)), "e0ddd7f2"),  # movk x0, #48879, lsl #32
    (movk, (w0, 0xffff), "e0ff9f72"),  # movk w0, #65535
    (mov, (x0, x1), "e00301aa"),  # mov x0, x1
    (mov, (w0, wzr), "e0031f2a"),  # mov w0, wzr
    (mov, (x0, 0), "00008052"),  # mov x0, #0
    (mov, (x0, 0xffff), "e0ff9f52"),  # mov x0, #65535
    (mov, (w0, 0x10000), "e0031032"),  # mov w0, #65536
    (mov, (x0, 0xff00ff00ff00ff00), "e09f08b2"),  # mov x0, #18374966859414961920
    (mov, (sp, x0), "1f000091"),  # mov sp, x0
    (mov, (x0, sp), "e0030091"),  # mov x0, sp
    (mov, (x0, x1 << 4), "e01301aa"),  # mov x0, x1, lsl #4
    (mvn, (x0, x1), "e00321aa"),  # mvn x0, x1
    (mvn, (w0, w1 >> 2), "e00b612a"),  # mvn w0, w1, lsr #2
    (csel, (x0, x1, x2, "eq"), "2000829a"),  # csel x0, x1, x2, eq
    (csel, (w0, w1, w2, "lo"), "2030821a"),  # csel w0, w1, w2, lo
    (csinc, (x0, x1, x2, "ne"), "2014829a"),  # csinc x0, x1, x2, ne
    (csinv, (x0, x1, x2, "ge"), "20a082da"),  # csinv x0, x1, x2, ge
    (csneg, (w0, w1, w2, "lt"), "20b4825a"),  # csneg w0, w1, w2, lt
    (cset, (x0, "eq"), "e0179f9a"),  # cset x0, eq
    (cset, (w0, "hs"), "e0379f1a"),  # cset w0, hs
    (csetm, (x0, "mi"), "e0539fda"),  # csetm x0, mi
    (cinc, (x0, x1, "gt"), "20d4819a"),  # cinc x0, x1, gt
    (cinv, (w0, w1, "le"), "20c0815a"),  # cinv w0, w1, le
    (cneg, (x0, x1, "vs"), "207481da"),  # cneg x0, x1, vs
    (ccmn, (x0, x1, 4, "ne"), "041041ba"),  # ccmn x0, x1, #4, ne
    (ccmn, (w0, 31, 15, "al"), "0fe85f3a"),  # ccmn w0, #31, #15, al
    (ccmp, (x0, x1, 0, "eq"), "000041fa"),  # ccmp x0, x1, #0, eq
    (ccmp, (x0, 5, 2, "hi"), "028845fa"),  # ccmp x0, #5, #2, hi
    (madd, (x0, x1, x2, x3), "200c029b"),  # madd x0, x1, x2, x3
    (msub, (w0, w1, w2, w3), "208c021b"),  # msub w0, w1, w2, w3
    (mul, (x0, x1, x2), "207c029b"),  # mul x0, x1, x2
    (mneg, (w0, w1, w2), "20fc021b"),  # mneg w0, w1, w2
    (smaddl, (x0, w1, w2, x3), "200c229b"),  # smaddl x0, w1, w2, x3
    (smsubl, (x0, w1, w2, x3), "208c229b"),  # smsubl x0, w1, w2, x3
    (smull, (x0, w1, w2), "207c229b"),  # smull x0, w1, w2
    (smnegl, (x0, w1, w2), "20fc229b"),  # smnegl x0, w1, w2
    (smulh, (x0, x1, x2), "207c429b"),  # smulh x0, x1, x2
    (umaddl, (x0, w1, w2, x3), "200ca29b"),  # umaddl x0, w1, w2, x3
    (umsubl, (x0, w1, w2, x3), "208ca29b"),  # umsubl x0, w1, w2, x3
    (umull, (x0, w1, w2), "207ca29b"),  # umull x0, w1, w2
    (umnegl, (x0, w1, w2), "20fca29b"),  # umnegl x0, w1, w2
    (umulh, (x0, x1, x2), "207cc29b"),  # umulh x0, x1, x2
    (udiv, (x0, x1, x2), "2008c29a"),  # udiv x0, x1, x2
    (sdiv, (w0, w1, w2), "200cc21a"),  # sdiv w0, w1, w2
    (sbfm, (x0, x1, 3, 7), "201c4393"),  # sbfm x0, x1, #3, #7
    (sbfm, (w0, w1, 31, 31), "207c1f13"),  # sbfm w0, w1, #31, #31
    (bfm, (x0, x1, 63, 0), "20007fb3"),  # bfm x0, x1, #63, #0
    (bfm, (w0, w1, 4, 2), "20080433"),  # bfm w0, w1, #4, #2
    (ubfm, (x0, x1, 8, 15), "203c48d3"),  # ubfm x0, x1, #8, #15
    (ubfm, (w0, w1, 0, 7), "201c0053"),  # ubfm w0, w1, #0, #7
    (extr, (x0, x1, x2, 63), "20fcc293"),  # extr x0, x1, x2, #63
    (extr, (w0, w1, w2, 5), "20148213"),  # extr w0, w1, w2, #5
    (sxtb, (w0, w1), "201c0013"),  # sxtb w0, w1
    (sxtb, (x0, x1), "201c4093"),  # sxtb x0, x1
    (sxth, (w0, w1), "203c0013"),  # sxth w0, w1
    (sxth, (x0, x1), "203c4093"),  # sxth x0, x1
    (sxtw, (x0, w1), "207c4093"),  # sxtw x0, w1
    (uxtb, (w0, w1), "201c0053"),  # uxtb w0, w1
    (uxth, (w0, w1), "203c0053"),  # uxth w0, w1
    (sbfx, (x0, x1, 4, 8), "202c4493"),  # sbfx x0, x1, #4, #8
    (bfxil, (w0, w1, 0, 32), "207c0033"),  # bfxil w0, w1, #0, #32
    (ubfx, (x0, x1, 60, 4), "20fc7cd3"),  # ubfx x0, x1, #60, #4
    (sbfiz, (w0, w1, 3, 5), "20101d13"),  # sbfiz w0, w1, #3, #5
    (bfi, (x0, x1, 16, 16), "203c70b3"),  # bfi x0, x1, #16, #16
    (ubfiz, (x0, x1, 63, 1), "200041d3"),  # ubfiz x0, x1, #63, #1
    (lsl, (x0, x1, x2), "2020c29a"),  # lsl x0, x1, x2
    (lsl, (x0, x1, 3), "20f07dd3"),  # lsl x0, x1, #3
    (lsl, (w0, w1, 31), "20000153"),  # lsl w0, w1, #31
    (lsl, (x0, x1, 0), "20fc40d3"),  # lsl x0, x1, #0
    (lsr, (x0, x1, x2), "2024c29a"),  # lsr x0, x1, x2
    (lsr, (w0, w1, 4), "207c0453"),  # lsr w0, w1, #4
    (lsr, (x0, x1, 63), "20fc7fd3"),  # lsr x0, x1, #63
    (asr, (w0, w1, w2), "2028c21a"),  # asr w0, w1, w2
    (asr, (x0, x1, 1), "20fc4193"),  # asr x0, x1, #1
    (ror, (x0, x1, x2), "202cc29a"),  # ror x0, x1, x2
    (ror, (w0, w1, 7), "201c8113"),  # ror w0, w1, #7
    (ror, (x0, x1, 40), "20a0c193"),  # ror x0, x1, #40
    (clz, (x0, x1), "2010c0da"),  # clz x0, x1
    (cls, (w0, w1), "2014c05a"),  # cls w0, w1
    (rbit, (x0, x1), "2000c0da"),  # rbit x0, x1
    (rev, (w0, w1), "2008c05a"),  # rev w0, w1
    (rev, (x0, x1), "200cc0da"),  # rev x0, x1
    (rev16, (w0, w1), "2004c05a"),  # rev16 w0, w1
    (rev32, (x0, x1), "2008c0da"),  # rev32 x0, x1
    (strb, (w0, mem[x1]), "20000039"),  # strb w0, [x1]
    (strb, (w0, mem[x1 + 4095]), "20fc3f39"),  # strb w0, [x1, #4095]
    (strb, (w0, mem[x1 - 1]), "20f01f38"),  # strb w0, [x1, #-1]
    (ldrb, (w0, mem[x1 + x2]), "20686238"),  # ldrb w0, [x1, x2]
    (ldrb, (w0, mem[x1 + w2.uxtw()]), "20486238"),  # ldrb w0, [x1, w2, uxtw]
    (ldrsb, (w0, mem[x1 + 1]), "2004c039"),  # ldrsb w0, [x1, #1]
    (ldrsb, (x0, mem.post[x1, 1]), "20148038"),  # ldrsb x0, [x1], #1
    (strh, (w0, mem[sp + 2]), "e0070079"),  # strh w0, [sp, #2]
    (strh, (w0, mem[x1 + 3]), "20300078"),  # strh w0, [x1, #3]
    (ldrh, (w0, mem[x1 + (x2 << 1)]), "20786278"),  # ldrh w0, [x1, x2, lsl #1]
    (ldrsh, (w0, mem[x1 + w2.sxtw(1)]), "20d8e278"),  # ldrsh w0, [x1, w2, sxtw #1]
    (ldrsh, (x0, mem.pre[x1 - 2]), "20ec9f78"),  # ldrsh x0, [x1, #-2]!
    (str_, (w0, mem[x1 + 16380]), "20fc3fb9"),  # str w0, [x1, #16380]
    (str_, (x0, mem[sp]), "e00300f9"),  # str x0, [sp]
    (str_, (x0, mem.pre[sp - 16]), "e00f1ff8"),  # str x0, [sp, #-16]!
    (str_, (x0, mem.post[sp, 16]), "e00701f8"),  # str x0, [sp], #16
    (str_, (s0, mem[x0 + 4]), "000400bd"),  # str s0, [x0, #4]
    (str_, (d0, mem[x0 + x1.lsl(3)]), "007821fc"),  # str d0, [x0, x1, lsl #3]
    (str_, (x0, mem[x1 + x2.sxtx(3)]), "20f822f8"),  # str x0, [x1, x2, sxtx #3]
    (str_, (x0, mem[x1 + x2.sxtx()]), "20e822f8"),  # str x0, [x1, x2, sxtx]
    (str_, (w0, mem[x1 + w2.sxtw(2)]), "20d822b8"),  # str w0, [x1, w2, sxtw #2]
    (ldr, (x0, mem[x1 + 8]), "200440f9"),  # ldr x0, [x1, #8]
    (ldr, (x0, mem[x1 + 32760]), "20fc7ff9"),  # ldr x0, [x1, #32760]
    (ldr, (x0, mem[x1 + 4]), "204040f8"),  # ldr x0, [x1, #4]
    (ldr, (x0, mem[x1 - 256]), "200050f8"),  # ldr x0, [x1, #-256]
    (ldr, (w0, mem[x1 + 255]), "20f04fb8"),  # ldr w0, [x1, #255]
    (ldr, (x0, mem[x1 + (x2 << 3)]), "207862f8"),  # ldr x0, [x1, x2, lsl #3]
    (ldr, (x0, mem[x1 + x2.lsl(0)]), "206862f8"),  # ldr x0, [x1, x2, lsl #0]
    (ldr, (s0, mem[x1 + 4]), "200440bd"),  # ldr s0, [x1, #4]
    (ldr, (d0, mem.pre[x1 + 8]), "208c40fc"),  # ldr d0, [x1, #8]!
    (ldr, (d31, mem.post[sp, -8]), "ff875ffc"),  # ldr d31, [sp], #-8
    (ldrsw, (x0, mem[x1 + 4]), "200480b9"),  # ldrsw x0, [x1, #4]
    (stp, (x29, x30, mem.pre[sp - 16]), "fd7bbfa9"),  # stp x29, x30, [sp, #-16]!
    (stp, (w0, w1, mem[x2 + 8]), "40040129"),  # stp w0, w1, [x2, #8]
    (stp, (s0, s1, mem.post[x2, 256 - 4]), "40849f2c"),  # stp s0, s1, [x2], #252
    (stp, (d8, d9, mem[sp + 504]), "e8a71f6d"),  # stp d8, d9, [sp, #504]
    (stp, (q0, q1, mem[x0 - 1024]), "000420ad"),  # stp q0, q1, [x0, #-1024]
    (ldp, (x29, x30, mem.post[sp, 16]), "fd7bc1a8"),  # ldp x29, x30, [sp], #16
    (ldp, (w0, w1, mem[x2]), "40044029"),  # ldp w0, w1, [x2]
    (ldp, (s0, s1, mem[x2 + 4]), "4084402d"),  # ldp s0, s1, [x2, #4]
    (ldp, (d0, d1, mem.pre[x2 - 512]), "4004e06d"),  # ldp d0, d1, [x2, #-512]!
    (ldp, (q0, q1, mem[x0 + 1008]), "00845fad"),  # ldp q0, q1, [x0, #1008]
    (ldpsw, (x0, x1, mem[x2 + 8]), "40044169"),  # ldpsw x0, x1, [x2, #8]
    (blr, (x0,), "00003fd6"),  # blr x0
    (br, (x16,), "00021fd6"),  # br x16
    (ret, (), "c0035fd6"),  # ret
    (ret, (x1,), "20005fd6"),  # ret x1
    (bti, ("c",), "5f2403d5"),  # bti c
    (bti, ("j",), "9f2403d5"),  # bti j
    (bti, ("jc",), "df2403d5"),  # bti jc
    (blraaz, (x0,), "1f083fd6"),  # blraaz x0
    (blrabz, (x1,), "3f0c3fd6"),  # blrabz x1
    (braa, (x0, x1), "01081fd7"),  # braa x0, x1
    (brab, (x2, x3), "430c1fd7"),  # brab x2, x3
    (braaz, (x4,), "9f081fd6"),  # braaz x4
    (brabz, (x5,), "bf0c1fd6"),  # brabz x5
    (paciasp, (), "3f2303d5"),  # paciasp
    (pacibsp, (), "7f2303d5"),  # pacibsp
    (autiasp, (), "bf2303d5"),  # autiasp
    (autibsp, (), "ff2303d5"),  # autibsp
    (retaa, (), "ff0b5fd6"),  # retaa
    (retab, (), "ff0f5fd6"),  # retab
    (nop, (), "1f2003d5"),  # nop
    (brk, (), "000020d4"),  # brk
    (brk, (0xf000,), "00003ed4"),  # brk #61440
    (fmov, (d0, d1), "2040601e"),  # fmov d0, d1
    (fmov, (s0, s1), "2040201e"),  # fmov s0, s1
    (fmov, (w0, s1), "2000261e"),  # fmov w0, s1
    (fmov, (s0, w1), "2000271e"),  # fmov s0, w1
    (fmov, (x0, d1), "2000669e"),  # fmov x0, d1
    (fmov, (d0, x1), "2000679e"),  # fmov d0, x1
    (fmov, (d0, 1.0), "00106e1e"),  # fmov d0, #1.0
    (fmov, (s0, -0.125), "0010381e"),  # fmov s0, #-0.125
    (fmov, (d0, 31.0), "00f0671e"),  # fmov d0, #31.0
    (fmov, (d0, 2), "0010601e"),  # fmov d0, #2
    (fabs, (d0, d1), "20c0601e"),  # fabs d0, d1
    (fneg, (s0, s1), "2040211e"),  # fneg s0, s1
    (fsqrt, (d0, d1), "20c0611e"),  # fsqrt d0, d1
    (fcvt, (d0, s1), "20c0221e"),  # fcvt d0, s1
    (fcvt, (s0, d1), "2040621e"),  # fcvt s0, d1
    (fcvtas, (w0, s1), "2000241e"),  # fcvtas w0, s1
    (fcvtau, (x0, d1), "2000659e"),  # fcvtau x0, d1
    (fcvtms, (w0, d1), "2000701e"),  # fcvtms w0, d1
    (fcvtmu, (x0, s1), "2000319e"),  # fcvtmu x0, s1
    (fcvtns, (w0, s1), "2000201e"),  # fcvtns w0, s1
    (fcvtnu, (x0, d1), "2000619e"),  # fcvtnu x0, d1
    (fcvtps, (w0, s1), "2000281e"),  # fcvtps w0, s1
    (fcvtpu, (x0, d1), "2000699e"),  # fcvtpu x0, d1
    (fcvtzs, (w0, d1), "2000781e"),  # fcvtzs w0, d1
    (fcvtzs, (x0, s1), "2000389e"),  # fcvtzs x0, s1
    (fcvtzu, (x0, d1), "2000799e"),  # fcvtzu x0, d1
    (scvtf, (s0, w1), "2000221e"),  # scvtf s0, w1
    (scvtf, (d0, x1), "2000629e"),  # scvtf d0, x1
    (ucvtf, (d0, w1), "2000631e"),  # ucvtf d0, w1
    (ucvtf, (s0, x1), "2000239e"),  # ucvtf s0, x1
    (frintn, (d0, d1), "2040641e"),  # frintn d0, d1
    (frintp, (s0, s1), "20c0241e"),  # frintp s0, s1
    (frintm, (d0, d1), "2040651e"),  # frintm d0, d1
    (frintz, (d0, d1), "20c0651e"),  # frintz d0, d1
    (frinta, (s0, s1), "2040261e"),  # frinta s0, s1
    (frintx, (d0, d1), "2040671e"),  # frintx d0, d1
    (frinti, (d0, d1), "20c0671e"),  # frinti d0, d1
    (fadd, (d0, d1, d2), "2028621e"),  # fadd d0, d1, d2
    (fsub, (s0, s1, s2), "2038221e"),  # fsub s0, s1, s2
    (fmul, (d0, d1, d2), "2008621e"),  # fmul d0, d1, d2
    (fnmul, (d0, d1, d2), "2088621e"),  # fnmul d0, d1, d2
    (fdiv, (s0, s1, s2), "2018221e"),  # fdiv s0, s1, s2
    (fmadd, (d0, d1, d2, d3), "200c421f"),  # fmadd d0, d1, d2, d3
    (fmsub, (s0, s1, s2, s3), "208c021f"),  # fmsub s0, s1, s2, s3
    (fnmadd, (d0, d1, d2, d3), "200c621f"),  # fnmadd d0, d1, d2, d3
    (fnmsub, (d0, d1, d2, d3), "208c621f"),  # fnmsub d0, d1, d2, d3
    (fmax, (d0, d1, d2), "2048621e"),  # fmax d0, d1, d2
    (fmaxnm, (d0, d1, d2), "2068621e"),  # fmaxnm d0, d1, d2
    (fmin, (s0, s1, s2), "2058221e"),  # fmin s0, s1, s2
    (fminnm, (d0, d1, d2), "2078621e"),  # fminnm d0, d1, d2
    (fcmp, (d0, d1), "0020611e"),  # fcmp d0, d1
    (fcmp, (s0, 0), "0820201e"),  # fcmp s0, #0
    (fcmp, (d0, 0.0), "0820601e"),  # fcmp d0, #0.0
    (fcmpe, (d0, d1), "1020611e"),  # fcmpe d0, d1
    (fcmpe, (s0, 0.0), "1820201e"),  # fcmpe s0, #0.0
    (fccmp, (d0, d1, 4, "ne"), "0414611e"),  # fccmp d0, d1, #4, ne
    (fccmpe, (s0, s1, 0, "eq"), "1004211e"),  # fccmpe s0, s1, #0, eq
    (fcsel, (d0, d1, d2, "gt"), "20cc621e"),  # fcsel d0, d1, d2, gt
]

# Branch and literal targets: (fn, ops, backward, forward). Backward: the
# label is bound one nop before the instruction (-4). Forward: the label
# follows the instruction and two nops (+12).
LABEL_CASES = [
    (adr, (x0, LBL), "e0ffff10", "60000010"),  # adr x0, >1
    (ldr, (x0, LBL), "e0ffff58", "60000058"),  # ldr x0, >1
    (ldr, (w0, LBL), "e0ffff18", "60000018"),  # ldr w0, >1
    (ldr, (s0, LBL), "e0ffff1c", "6000001c"),  # ldr s0, >1
    (ldr, (d0, LBL), "e0ffff5c", "6000005c"),  # ldr d0, >1
    (ldrsw, (x0, LBL), "e0ffff98", "60000098"),  # ldrsw x0, >1
    (b, (LBL,), "ffffff17", "03000014"),  # b >1
    (bl, (LBL,), "ffffff97", "03000094"),  # bl >1
    (cbz, (x0, LBL), "e0ffffb4", "600000b4"),  # cbz x0, >1
    (cbnz, (w0, LBL), "e0ffff35", "60000035"),  # cbnz w0, >1
    (tbz, (w0, 0, LBL), "e0ff0736", "60000036"),  # tbz w0, #0, >1
    (tbz, (x0, 63, LBL), "e0ffffb6", "6000f8b6"),  # tbz x0, #63, >1
    (tbnz, (x1, 32, LBL), "e1ff07b7", "610000b7"),  # tbnz x1, #32, >1
    (beq, (LBL,), "e0ffff54", "60000054"),  # beq >1
    (bne, (LBL,), "e1ffff54", "61000054"),  # bne >1
    (bhs, (LBL,), "e2ffff54", "62000054"),  # bhs >1
    (blo, (LBL,), "e3ffff54", "63000054"),  # blo >1
    (bal, (LBL,), "eeffff54", "6e000054"),  # bal >1
]
# fmt: on


def _case_id(case):
    fn, ops = case[0], case[1]
    return f"{fn.__name__} {', '.join(map(str, ops))}"


def _subst(ops, lbl):
    return [lbl if isinstance(o, str) and o == LBL else o for o in ops]


@pytest.mark.parametrize("fn, ops, want", CASES, ids=[_case_id(c) for c in CASES])
def test_encode(fn, ops, want):
    assert hexof(fn, *ops) == want


@pytest.mark.parametrize("fn, ops, back, fwd", LABEL_CASES, ids=[_case_id(c) for c in LABEL_CASES])
def test_label_backward(fn, ops, back, fwd):
    a = Assembler(A)
    lbl = a.label()
    a.nop()
    fn(*_subst(ops, lbl), asm=a)
    assert a.link().data[4:8].hex() == back


@pytest.mark.parametrize("fn, ops, back, fwd", LABEL_CASES, ids=[_case_id(c) for c in LABEL_CASES])
def test_label_forward(fn, ops, back, fwd):
    a = Assembler(A)
    lbl = Label()
    fn(*_subst(ops, lbl), asm=a)
    a.nop()
    a.nop()
    a.label(lbl)
    assert a.link().data[:4].hex() == fwd


def test_every_mnemonic_is_tested():
    tested = {c[0].__name__ for c in CASES + LABEL_CASES} | {"adrp"}
    names = {k.rpartition("_")[0] for k in MAP_OP} - {"b" + c for c in MAP_COND}
    py = {"and": "and_", "str": "str_"}
    assert {py.get(n, n) for n in names} - tested == set()


# -- labels, conditions, externs ------------------------------------------------


def test_b_cond_attributes():
    for cond in MAP_COND:
        a1, a2 = Assembler(A), Assembler(A)
        getattr(b, cond)("x", asm=a1)  # noqa: F405
        INSNS["b" + cond]("x", asm=a2)
        a1.label("x")
        a2.label("x")
        assert a1.link().data == a2.link().data
    assert b.eq.__name__ == "eq"  # noqa: F405


def test_string_labels_and_methods():
    a = Assembler(A)
    a.label("top")
    a.subs(x0, x0, 1)  # noqa: F405
    a.b.ne("top")
    a.cbz(x0, "out")  # noqa: F405
    a.b("top")
    a.label("out")
    a.ret()
    assert a.link().data.hex() == "000400f1e1ffff54400000b4fdffff17c0035fd6"


def test_adrp_page_delta():
    # adrp takes the 4KB page delta of the addresses, not the byte delta
    # shifted right by 12: at 0xffc a label at 0x1004 is one page ahead.
    a = Assembler(A)
    lbl = Label()
    for _ in range(1023):
        a.nop()
    adrp(x0, lbl, asm=a)  # noqa: F405
    a.nop()
    a.label(lbl)
    assert a.link().data[1023 * 4 : 1024 * 4].hex() == "000000b0"
    assert a.link(base=0x1000).data[1023 * 4 : 1024 * 4].hex() == "000000b0"
    # Backward within the same page: delta 0.
    a = Assembler(A)
    lbl = a.label()
    a.nop()
    adrp(x0, lbl, asm=a)  # noqa: F405
    assert a.link().data[4:].hex() == "00000090"
    assert a.cur.patches[0].kind is REL21_ADRP


def test_extern_targets():
    a = Assembler(A)
    bl(Extern("f"), asm=a)  # noqa: F405
    b(Extern("g", 0x100000), asm=a)  # noqa: F405
    adr(x1, Extern("h"), asm=a)  # noqa: F405
    img = a.link(base=0x10000, externs={"f": 0x10000 + 0x800, "h": 0x10000 + 8 + 3})
    assert img.data.hex() == "00020094" + "ffbf0314" + "01000070"
    with pytest.raises(LinkError, match="multiple of 4"):
        a.link(base=0x10000, externs={"f": 0x10002, "h": 0})
    with pytest.raises(LinkError, match="out of range"):
        a.link(base=0x10000, externs={"f": 0x10000 + (1 << 27), "h": 0x10000})


@pytest.mark.parametrize(
    "fn, ops, limit",
    [
        (bl, (), 1 << 27),  # noqa: F405
        (cbz, (x0,), 1 << 20),  # noqa: F405
        (ldr, (x0,), 1 << 20),  # noqa: F405
        (tbz, (x0, 1), 1 << 15),  # noqa: F405
        (adr, (x0,), 1 << 20),  # noqa: F405
    ],
)
def test_branch_ranges(fn, ops, limit):
    base = 1 << 32
    step = 1 if fn is adr else 4  # noqa: F405
    a = Assembler(A)
    fn(*ops, Extern("t"), asm=a)
    a.link(base=base, externs={"t": base + limit - step})
    a.link(base=base, externs={"t": base - limit})
    with pytest.raises(LinkError, match="out of range"):
        a.link(base=base, externs={"t": base + limit})
    with pytest.raises(LinkError, match="out of range"):
        a.link(base=base, externs={"t": base - limit - step})


def test_patch_kinds_or_into_word():
    buf = bytearray(bytes.fromhex("00000094"))
    REL26.apply(buf, 0, 0x1008, 0x1000)
    assert buf.hex() == "02000094"
    buf = bytearray(bytes.fromhex("00000054"))
    REL19.apply(buf, 0, 0x1000 - 4, 0x1000)
    assert buf.hex() == "e0ffff54"
    buf = bytearray(bytes.fromhex("00000036"))
    REL14.apply(buf, 0, 0x1000 + 0x7FFC, 0x1000)
    assert buf.hex() == "e0ff0336"
    buf = bytearray(bytes.fromhex("00000010"))
    REL21_ADR.apply(buf, 0, 0x1000 + 5, 0x1000)
    assert buf.hex() == "20000030"
    buf = bytearray(bytes.fromhex("00000090"))
    REL21_ADRP.apply(buf, 0, 0x5000, 0x1FFC)
    assert buf.hex() == "20000090"
    for kind in (REL26, REL19, REL14, REL21_ADR, REL21_ADRP):
        assert kind.size == 4


def test_core_accepts_arch_patch_kinds():
    # The core stores and applies any PatchKind; nothing is aarch64 specific.
    a = Assembler(A)
    lbl = Label("t")
    a.emit(bytes.fromhex("1f2003d5"))
    a.emit_patch(REL19, lbl)
    a.cur.buf[-4:] = bytes.fromhex("00000054")  # b.eq, field left zero
    a.label(lbl)
    img = a.link(base=0x4000)
    assert img.data.hex() == "1f2003d5" + "20000054"
    assert img.symbols == {"t": 8}


def test_data_directives_with_labels():
    a = Assembler(A)
    lbl = a.label("here")
    a.qword(lbl)
    a.dword(0x12345678)
    assert a.link(base=0x10000).data.hex() == "0000010000000000" + "78563412"


# -- arch object ---------------------------------------------------------------


def test_nop_fill_and_align():
    assert A.ARCH.nop_fill(8) == A.NOP * 2
    assert A.ARCH.nop_fill(6) == b"\0\0" + A.NOP
    assert len(A.ARCH.nop_fill(0)) == 0
    a = Assembler(A)
    a.ret()
    a.align(16)
    assert bytes(a.cur.buf) == bytes.fromhex("c0035fd6") + A.NOP * 3


def test_arch_attributes():
    assert A.ARCH.name == "aarch64" and A.ARCH.pointer_size == 8
    assert A.ARCH.insns is INSNS
    assert Assembler(A).arch is A.ARCH and Assembler(A.ARCH).arch is A.ARCH


def test_icache_flush(monkeypatch):
    calls = []

    def fake(start, end):
        calls.append((start.value, end.value))

    arch = A.Aarch64Arch()
    monkeypatch.setattr(A, "_find_clear_cache", lambda: fake)
    monkeypatch.setattr(A.platform, "machine", lambda: "x86_64")
    arch.icache_flush(0x1000, 64)
    assert calls == []
    monkeypatch.setattr(A.platform, "machine", lambda: "aarch64")
    arch.icache_flush(0x1000, 64)
    arch.icache_flush(0x2000, 0)
    assert calls == [(0x1000, 0x1040)]


@pytest.mark.parametrize("machine", ["aarch64", "arm64", "ARM64"])
def test_host_arch(monkeypatch, machine):
    import jita.core.assembler as asm_mod

    monkeypatch.setattr(asm_mod.platform, "machine", lambda: machine)
    assert Assembler().arch is A.ARCH


def test_star_exports():
    names = set(A.__all__)
    assert {"str_", "and_", "b", "beq", "mem", "sp", "xzr", "lr", "fp", "ARCH", "Aarch64Arch", "label"} <= names
    assert not {"str", "and", "encoder", "ctypes", "platform", "INSNS"} & names
    assert A.str_ is INSNS["str"]


def test_listing():
    from jita.tools.listing import listing

    a = Assembler(A)
    with a:
        label("loop")
        ldr(x1, mem.post[x0, 8])  # noqa: F405
        add(x2, x2, x1 << 1)  # noqa: F405
        movk(x3, 0xBEEF, lsl=16)  # noqa: F405
        b.ne("loop")  # noqa: F405
    text = listing(a)
    assert "ldr x1, [x0], #8" in text
    assert "add x2, x2, x1, lsl #1" in text
    assert "movk x3, 0xbeef, lsl #16" in text
    assert "b.ne loop" in text and "rel19 -> loop" in text


# -- operands ------------------------------------------------------------------


def test_register_objects():
    assert gp64(31) is xzr and gp32(31) is wzr and gp64(3) is x3  # noqa: F405
    assert fp32(1) is s1 and fp64(2) is d2 and fp128(3) is q3  # noqa: F405
    assert lr is x30 and fp is x29  # noqa: F405
    assert sp.kind == "sp" and sp.code == 31  # noqa: F405
    with pytest.raises(EncodeError):
        gp64(32)  # noqa: F405
    assert str(x1 << 3) == "x1, lsl #3"  # noqa: F405
    assert str(w1.uxtw()) == "w1, uxtw"  # noqa: F405
    assert str(x1.asr(2)) == "x1, asr #2"  # noqa: F405


def test_memory_operands():
    assert str(mem[x0]) == "[x0]"  # noqa: F405
    assert str(mem[sp + 16]) == "[sp, #16]"  # noqa: F405
    assert str(mem[x0 - 8]) == "[x0, #-8]"  # noqa: F405
    assert str(mem[x0 + x1]) == "[x0, x1]"  # noqa: F405
    assert str(mem[x0 + (x1 << 3)]) == "[x0, x1, lsl #3]"  # noqa: F405
    assert str(mem[x0 + w1.sxtw(2)]) == "[x0, w1, sxtw #2]"  # noqa: F405
    assert str(mem.pre[x0 + 16]) == "[x0, #16]!"  # noqa: F405
    assert str(mem.post[x0, -16]) == "[x0], #-16"  # noqa: F405
    assert mem[x0 + 8] == mem[x0 + 8]  # noqa: F405


@pytest.mark.parametrize(
    "build, match",
    [
        (lambda: mem[xzr], "memory base"),  # noqa: F405
        (lambda: mem[w0], "memory base"),  # noqa: F405
        (lambda: mem[d0], "memory base"),  # noqa: F405
        (lambda: mem[x0 + w1], "uxtw or sxtw"),  # noqa: F405
        (lambda: mem[x0 + x1.uxtw()], "lsl or sxtx"),  # noqa: F405
        (lambda: mem[x0 + x1 + 8], "index register and an offset"),  # noqa: F405
        (lambda: mem[x0 + x1 + x2], "too many registers"),  # noqa: F405
        (lambda: mem.pre[x0 + x1], "takes an immediate"),  # noqa: F405
        (lambda: mem.post[x0 + 8], "mem.post"),  # noqa: F405
        (lambda: x0 + x1 << 3, "index << n"),  # noqa: F405
        (lambda: mem[x0 + sp], "memory index"),  # noqa: F405
        (lambda: mem[8], "base register"),  # noqa: F405
    ],
)
def test_memory_operand_errors(build, match):
    with pytest.raises(EncodeError, match=match):
        build()


# -- encoder errors ------------------------------------------------------------


@pytest.mark.parametrize(
    "fn, ops, match",
    [
        (add, (x0, x1, 5000), "12 bit unsigned"),  # noqa: F405
        (add, (x0, x1, 0x100000000), "out of range"),  # noqa: F405
        (add, (x0, w1, x2), "size mismatch"),  # noqa: F405
        (add, (x0, xzr, 1), "register 31 means sp"),  # noqa: F405
        (add, (x0, xzr, x1.uxtx()), "register 31 means sp"),  # noqa: F405
        (add, (xzr, x1, w2.uxtw()), "register 31 means sp"),  # noqa: F405
        (orr, (xzr, x0, 1), "register 31 means sp"),  # noqa: F405
        (mov, (sp, xzr), "register 31 means sp"),  # noqa: F405
        (add, (x0, x1, sp), "sp is not allowed"),  # noqa: F405
        (add, (w0, w1, w2 << 32), "32 bit register"),  # noqa: F405
        (add, (x0, x1, x2 << 64), "out of range"),  # noqa: F405
        (add, (x0, x1, x2.uxtx(5)), "0..4"),  # noqa: F405
        (add, (x0, x1, w2.uxtx()), "write the x register"),  # noqa: F405
        (add, (sp, x1, w2.sxtx(3)), "write the x register"),  # noqa: F405
        (sub, (x1, sp, w2 << 7), "0..4"),  # noqa: F405
        (and_, (x0, x1, 0), "logical"),  # noqa: F405
        (and_, (w0, w1, 0x100000000), "logical"),  # noqa: F405
        (mov, (x0, 0x12345), "out of range"),  # noqa: F405
        (mov, (d0, 1), "register type"),  # noqa: F405
        (movz, (x0, 1, Mod("lsl", 8)), "0, 16, 32 or 48"),  # noqa: F405
        (movz, (w0, 1, Mod("lsl", 32)), "0 or 16"),  # noqa: F405
        (lsr, (w0, w1, 32), "32 bit register"),  # noqa: F405
        (ubfm, (w0, w1, 0, 40), "32 bit register"),  # noqa: F405
        (extr, (w0, w1, w2, 32), "32 bit register"),  # noqa: F405
        (ubfx, (x0, x1, 60, 8), "out of range"),  # noqa: F405
        (ubfx, (x0, x1, 8, 0), "out of range"),  # noqa: F405
        (bfi, (w0, w1, 30, 4), "out of range"),  # noqa: F405
        (lsl, (x0, x1, 64), "out of range"),  # noqa: F405
        (tbz, (w0, 32, "x"), "32 bit register"),  # noqa: F405
        (tbz, (x0, 64, "x"), "0..63"),  # noqa: F405
        (cset, (x0, "al"), "inverted to nv"),  # noqa: F405
        (cinc, (x0, x1, "al"), "inverted to nv"),  # noqa: F405
        (csel, (x0, x1, x2, "xx"), "expected a condition"),  # noqa: F405
        (braa, (x0, xzr), "modifier"),  # noqa: F405
        (ldr, (x0, mem[x1 + 32761]), "out of range"),  # noqa: F405
        (ldr, (x0, mem[x1 + (x2 << 2)]), "must be 0 or 3"),  # noqa: F405
        (ldr, (x0, mem.pre[x1 + 256]), "signed 9 bit"),  # noqa: F405
        (ldr, (x1, mem.pre[x1 + 8]), "writeback"),  # noqa: F405
        (str_, (x1, mem.post[x1, 8]), "writeback"),  # noqa: F405
        (ldp, (x0, x0, mem[sp]), "twice"),  # noqa: F405
        (ldp, (x0, x1, mem.pre[x1 + 16]), "writeback"),  # noqa: F405
        (stp, (x0, x1, mem[sp + 4]), "multiple of 8"),  # noqa: F405
        (stp, (x0, x1, mem[sp + x2]), "not an index"),  # noqa: F405
        (ldr, (x0, x1 + 8), "mem\\[x1 \\+ 8\\]"),  # noqa: F405
        (ldr, (x0, mem[x1], 8), "too many operands"),  # noqa: F405
        (fmov, (d0, 0.1), "floating point immediate"),  # noqa: F405
        (fmov, (d0, 0.0), "floating point immediate"),  # noqa: F405
        (fcmp, (d0, 1.0), "constant 0"),  # noqa: F405
        (b, (x0,), "expected a label"),  # noqa: F405
        (b, (5,), "expected a label"),  # noqa: F405
        (bti, ("x",), "bti target"),  # noqa: F405
        (cset, (x0,), "expects 2 operands"),  # noqa: F405
        (ret, (x0, x1), "expects 0 or 1 operands"),  # noqa: F405
    ],
    ids=lambda v: v.__name__ if callable(v) else None,
)
def test_encode_errors(fn, ops, match):
    a = Assembler(A)
    with pytest.raises(EncodeError, match=match):
        fn(*ops, asm=a)
    assert a.pos() == 0 and not a.cur.patches


def test_unknown_instruction():
    from jita.aarch64.encoder import encode as raw

    with pytest.raises(EncodeError, match="unknown instruction"):
        raw(Assembler(A), "frobnicate", ())


def test_logical_immediates_beyond_dynasm_literals():
    # Every encodable bitmask is accepted, like DynASM's runtime path.
    assert hexof(and_, x0, x1, 0xFFFF0000FFFF0000) == "203c1092"  # noqa: F405
    assert hexof(and_, x0, x1, -2) == hexof(and_, x0, x1, 0xFFFFFFFFFFFFFFFE)  # noqa: F405
    assert hexof(and_, w0, w1, -2) == hexof(and_, w0, w1, 0xFFFFFFFE)  # noqa: F405
    assert hexof(mov, x0, 0xAAAAAAAAAAAAAAAA) == "e0f301b2"  # noqa: F405


# -- oracle --------------------------------------------------------------------

# Cases where DynASM's encoding differs from the GNU assembler's (equivalent
# instruction) or where DynASM accepts a spelling GNU syntax does not.
DIVERGENT = {
    "mov x0, #0": "DynASM uses movz w0 (zero-extends), GNU movz x0",
    "mov x0, #65535": "DynASM uses movz w0 (zero-extends), GNU movz x0",
    "mov w0, #65536": "DynASM uses orr with a bitmask, GNU movz w0, #1, lsl #16",
}

_ALIAS_TEXT = {"and_": "and", "str_": "str"}


def gnu_text(fn, ops):
    name = _ALIAS_TEXT.get(fn.__name__, fn.__name__)
    if name.startswith("b") and name[1:] in MAP_COND:
        name = "b." + name[1:]

    def one(o):
        if isinstance(o, float):
            return f"#{o!r}"
        if isinstance(o, int):
            return f"#{o}"
        return str(o)

    if name in ("fcmp", "fcmpe") and not isinstance(ops[1], Reg):  # noqa: F405
        return f"{name} {ops[0]}, #0.0"
    if name == "brk" and not ops:
        return "brk #0"
    if name in ("mov", "mvn") and len(ops) == 2 and isinstance(ops[1], RegMod):  # noqa: F405
        zr = "xzr" if ops[0].rt == "x" else "wzr"
        return f"{'orr' if name == 'mov' else 'orn'} {ops[0]}, {zr}, {ops[1]}"
    return name + (" " + ", ".join(one(o) for o in ops) if ops else "")


@requires_aarch64_oracle
@pytest.mark.parametrize("fn, ops, want", CASES, ids=[_case_id(c) for c in CASES])
def test_oracle(fn, ops, want):
    text = gnu_text(fn, ops)
    try:
        got = assemble(text, arch="aarch64").hex()
    except OracleError:
        assert text in DIVERGENT, f"{aarch64_assembler()[0]} rejects {text!r}"
        return
    if got != want:
        assert text in DIVERGENT, f"{text!r}: jita/DynASM {want}, {aarch64_assembler()[0]} {got}"


@requires_aarch64_oracle
@pytest.mark.parametrize("fn, ops, back, fwd", LABEL_CASES, ids=[_case_id(c) for c in LABEL_CASES])
def test_oracle_labels(fn, ops, back, fwd):
    text = gnu_text(fn, [o for o in ops])
    code = assemble("1:\nnop\n" + text.replace(LBL, "1b"), arch="aarch64")
    assert code[4:8].hex() == back
    code = assemble(text.replace(LBL, "1f") + "\nnop\nnop\n1:", arch="aarch64")
    assert code[:4].hex() == fwd
