"""riscv64 encoder tests.

`CASES`, `RM_CASES` and `LABEL_CASES` hold byte-exact expectations
produced by GNU as (riscv64-linux-gnu-as with `.option norvc` and
`.option norelax`) for the instruction in the comment, so they run
without binutils. The oracle tests check every case against GNU as again
when it is installed, and add checks too broad to list: every register
number, and `li` over many random values.
"""

import platform
import random

import pytest
from oracle import OracleError, assemble, requires_riscv64_oracle, riscv64_disassemble

import jita.riscv64 as R
from jita import Assembler, EncodeError, Extern, Label, LinkError
from jita.riscv64 import *  # noqa: F403
from jita.riscv64.encoder import MNEMONIC_ARGC, li_words
from jita.riscv64.insns import INSNS, MNEMONICS, PY_NAMES, Group
from jita.riscv64.mem import MemExpr
from jita.riscv64.patch import B12, J20, PCREL32, split_pcrel

LBL = "LBL"  # replaced by a Label in LABEL_CASES


def encode(fn, *ops, **kw):
    """Encode one instruction into a fresh Assembler, return (bytes, patches)."""
    a = Assembler(R)
    fn(*ops, asm=a, **kw)
    return bytes(a.cur.buf), a.cur.patches


def hexof(fn, *ops, **kw):
    return encode(fn, *ops, **kw)[0].hex()


# fmt: off
CASES = [
    (lui, (a0, 0x12345), "37553412"),  # lui a0, 74565
    (lui, (t6, 0xfffff), "b7ffffff"),  # lui t6, 1048575
    (lui, (a0, 0), "37050000"),  # lui a0, 0
    (auipc, (a0, 0x80000), "17050080"),  # auipc a0, 524288
    (auipc, (ra, 1), "97100000"),  # auipc ra, 1
    (jalr, (a0,), "e7000500"),  # jalr a0
    (jalr, (a0, a1), "67850500"),  # jalr a0, a1
    (jalr, (a0, mem[a1 + 12]), "6785c500"),  # jalr a0, 12(a1)
    (jalr, (zero, mem[ra - 2048]), "67800080"),  # jalr zero, -2048(ra)
    (jalr, (a0, a1, 2047), "6785f57f"),  # jalr a0, a1, 2047
    (addi, (a0, a1, -2048), "13850580"),  # addi a0, a1, -2048
    (addi, (a0, a1, 2047), "1385f57f"),  # addi a0, a1, 2047
    (addi, (sp, sp, -16), "130101ff"),  # addi sp, sp, -16
    (slti, (a0, a1, -1), "13a5f5ff"),  # slti a0, a1, -1
    (sltiu, (a0, a1, 1), "13b51500"),  # sltiu a0, a1, 1
    (xori, (a0, a1, -1), "13c5f5ff"),  # xori a0, a1, -1
    (ori, (t0, t1, 0x7ff), "9362f37f"),  # ori t0, t1, 2047
    (andi, (s0, s1, 0xff), "13f4f40f"),  # andi s0, s1, 255
    (slli, (a0, a1, 0), "13950500"),  # slli a0, a1, 0
    (slli, (a0, a1, 63), "1395f503"),  # slli a0, a1, 63
    (srli, (a0, a1, 32), "13d50502"),  # srli a0, a1, 32
    (srai, (a0, a1, 63), "13d5f543"),  # srai a0, a1, 63
    (add, (a0, a1, a2), "3385c500"),  # add a0, a1, a2
    (sub, (t3, t4, t5), "338eee41"),  # sub t3, t4, t5
    (sll, (a0, a1, a2), "3395c500"),  # sll a0, a1, a2
    (slt, (a0, a1, a2), "33a5c500"),  # slt a0, a1, a2
    (sltu, (a0, a1, a2), "33b5c500"),  # sltu a0, a1, a2
    (xor, (a0, a1, a2), "33c5c500"),  # xor a0, a1, a2
    (srl, (a0, a1, a2), "33d5c500"),  # srl a0, a1, a2
    (sra, (a0, a1, a2), "33d5c540"),  # sra a0, a1, a2
    (or_, (a0, a1, a2), "33e5c500"),  # or a0, a1, a2
    (and_, (a0, a1, a2), "33f5c500"),  # and a0, a1, a2
    (addiw, (a0, a1, -2048), "1b850580"),  # addiw a0, a1, -2048
    (addiw, (a0, a1, 2047), "1b85f57f"),  # addiw a0, a1, 2047
    (slliw, (a0, a1, 31), "1b95f501"),  # slliw a0, a1, 31
    (srliw, (a0, a1, 0), "1bd50500"),  # srliw a0, a1, 0
    (sraiw, (a0, a1, 31), "1bd5f541"),  # sraiw a0, a1, 31
    (addw, (a0, a1, a2), "3b85c500"),  # addw a0, a1, a2
    (subw, (a0, a1, a2), "3b85c540"),  # subw a0, a1, a2
    (sllw, (a0, a1, a2), "3b95c500"),  # sllw a0, a1, a2
    (srlw, (a0, a1, a2), "3bd5c500"),  # srlw a0, a1, a2
    (sraw, (a0, a1, a2), "3bd5c540"),  # sraw a0, a1, a2
    (lb, (a0, mem[a1]), "03850500"),  # lb a0, 0(a1)
    (lh, (a0, mem[a1 + 2]), "03952500"),  # lh a0, 2(a1)
    (lw, (a0, mem[sp + 2047]), "0325f17f"),  # lw a0, 2047(sp)
    (ld, (a0, mem[sp - 2048]), "03350180"),  # ld a0, -2048(sp)
    (lbu, (a0, mem[a1 - 1]), "03c5f5ff"),  # lbu a0, -1(a1)
    (lhu, (a0, mem[a1 + 0x7fe]), "03d5e57f"),  # lhu a0, 2046(a1)
    (lwu, (a0, mem[t6 + 4]), "03e54f00"),  # lwu a0, 4(t6)
    (sb, (a0, mem[a1]), "2380a500"),  # sb a0, 0(a1)
    (sh, (a0, mem[a1 - 2]), "239fa5fe"),  # sh a0, -2(a1)
    (sw, (a0, mem[sp + 2047]), "a32fa17e"),  # sw a0, 2047(sp)
    (sd, (ra, mem[sp - 2048]), "23301180"),  # sd ra, -2048(sp)
    (sd, (s11, mem[t0 + 0x20]), "23b0b203"),  # sd s11, 32(t0)
    (ecall, (), "73000000"),  # ecall
    (ebreak, (), "73001000"),  # ebreak
    (fence, (), "0f00f00f"),  # fence
    (fence, ("rw", "rw"), "0f003003"),  # fence rw, rw
    (fence, ("iorw", "w"), "0f00100f"),  # fence iorw, w
    (fence, ("r", "iorw"), "0f00f002"),  # fence r, iorw
    (fence, ("io", "or"), "0f00600c"),  # fence io, or
    (fence.tso, (), "0f003083"),  # fence.tso
    (fence.i, (), "0f100000"),  # fence.i
    (csrrw, (a0, 0xfff, a1), "7395f5ff"),  # csrrw a0, 4095, a1
    (csrrw, (zero, "fcsr", a1), "73903500"),  # csrrw zero, fcsr, a1
    (csrrs, (a0, "cycle", zero), "732500c0"),  # csrrs a0, cycle, zero
    (csrrc, (a0, "fflags", a2), "73351600"),  # csrrc a0, fflags, a2
    (csrrwi, (a0, "frm", 31), "73d52f00"),  # csrrwi a0, frm, 31
    (csrrsi, (a0, 0x7c0, 0), "7365007c"),  # csrrsi a0, 1984, 0
    (csrrci, (a0, "fflags", 5), "73f51200"),  # csrrci a0, fflags, 5
    (mul, (a0, a1, a2), "3385c502"),  # mul a0, a1, a2
    (mulh, (a0, a1, a2), "3395c502"),  # mulh a0, a1, a2
    (mulhsu, (a0, a1, a2), "33a5c502"),  # mulhsu a0, a1, a2
    (mulhu, (a0, a1, a2), "33b5c502"),  # mulhu a0, a1, a2
    (div, (a0, a1, a2), "33c5c502"),  # div a0, a1, a2
    (divu, (a0, a1, a2), "33d5c502"),  # divu a0, a1, a2
    (rem, (a0, a1, a2), "33e5c502"),  # rem a0, a1, a2
    (remu, (a0, a1, a2), "33f5c502"),  # remu a0, a1, a2
    (mulw, (a0, a1, a2), "3b85c502"),  # mulw a0, a1, a2
    (divw, (a0, a1, a2), "3bc5c502"),  # divw a0, a1, a2
    (divuw, (a0, a1, a2), "3bd5c502"),  # divuw a0, a1, a2
    (remw, (a0, a1, a2), "3be5c502"),  # remw a0, a1, a2
    (remuw, (a0, a1, a2), "3bf5c502"),  # remuw a0, a1, a2
    (flw, (fa0, mem[a1 + 4]), "07a54500"),  # flw fa0, 4(a1)
    (fsw, (fa0, mem[sp - 4]), "272ea1fe"),  # fsw fa0, -4(sp)
    (fld, (ft11, mem[sp + 2040]), "873f817f"),  # fld ft11, 2040(sp)
    (fsd, (fs11, mem[a0 - 2048]), "2730b581"),  # fsd fs11, -2048(a0)
    (fadd.s, (fa0, fa1, fa2), "53f5c500"),  # fadd.s fa0, fa1, fa2
    (fsub.s, (fa0, fa1, fa2), "53f5c508"),  # fsub.s fa0, fa1, fa2
    (fmul.s, (fa0, fa1, fa2), "53f5c510"),  # fmul.s fa0, fa1, fa2
    (fdiv.s, (fa0, fa1, fa2), "53f5c518"),  # fdiv.s fa0, fa1, fa2
    (fsqrt.s, (fa0, fa1), "53f50558"),  # fsqrt.s fa0, fa1
    (fsgnj.s, (fa0, fa1, fa2), "5385c520"),  # fsgnj.s fa0, fa1, fa2
    (fsgnjn.s, (fa0, fa1, fa2), "5395c520"),  # fsgnjn.s fa0, fa1, fa2
    (fsgnjx.s, (fa0, fa1, fa2), "53a5c520"),  # fsgnjx.s fa0, fa1, fa2
    (fmin.s, (fa0, fa1, fa2), "5385c528"),  # fmin.s fa0, fa1, fa2
    (fmax.s, (fa0, fa1, fa2), "5395c528"),  # fmax.s fa0, fa1, fa2
    (fmadd.s, (fa0, fa1, fa2, fa3), "43f5c568"),  # fmadd.s fa0, fa1, fa2, fa3
    (fmsub.s, (fa0, fa1, fa2, fa3), "47f5c568"),  # fmsub.s fa0, fa1, fa2, fa3
    (fnmsub.s, (fa0, fa1, fa2, fa3), "4bf5c568"),  # fnmsub.s fa0, fa1, fa2, fa3
    (fnmadd.s, (fa0, fa1, fa2, fa3), "4ff5c568"),  # fnmadd.s fa0, fa1, fa2, fa3
    (fadd.d, (fa0, fa1, fa2), "53f5c502"),  # fadd.d fa0, fa1, fa2
    (fsub.d, (ft0, ft1, ft2), "53f0200a"),  # fsub.d ft0, ft1, ft2
    (fmul.d, (fa0, fa1, fa2), "53f5c512"),  # fmul.d fa0, fa1, fa2
    (fdiv.d, (fa0, fa1, fa2), "53f5c51a"),  # fdiv.d fa0, fa1, fa2
    (fsqrt.d, (fa0, fa1), "53f5055a"),  # fsqrt.d fa0, fa1
    (fsgnj.d, (fa0, fa1, fa2), "5385c522"),  # fsgnj.d fa0, fa1, fa2
    (fsgnjn.d, (fa0, fa1, fa2), "5395c522"),  # fsgnjn.d fa0, fa1, fa2
    (fsgnjx.d, (fa0, fa1, fa2), "53a5c522"),  # fsgnjx.d fa0, fa1, fa2
    (fmin.d, (fa0, fa1, fa2), "5385c52a"),  # fmin.d fa0, fa1, fa2
    (fmax.d, (fa0, fa1, fa2), "5395c52a"),  # fmax.d fa0, fa1, fa2
    (fmadd.d, (fa0, fa1, fa2, fa3), "43f5c56a"),  # fmadd.d fa0, fa1, fa2, fa3
    (fmsub.d, (fa0, fa1, fa2, fa3), "47f5c56a"),  # fmsub.d fa0, fa1, fa2, fa3
    (fnmsub.d, (fa0, fa1, fa2, fa3), "4bf5c56a"),  # fnmsub.d fa0, fa1, fa2, fa3
    (fnmadd.d, (ft11, fs11, fa7, ft0), "cfff1d03"),  # fnmadd.d ft11, fs11, fa7, ft0
    (feq.s, (a0, fa0, fa1), "5325b5a0"),  # feq.s a0, fa0, fa1
    (flt.s, (a0, fa0, fa1), "5315b5a0"),  # flt.s a0, fa0, fa1
    (fle.s, (a0, fa0, fa1), "5305b5a0"),  # fle.s a0, fa0, fa1
    (feq.d, (a0, fa0, fa1), "5325b5a2"),  # feq.d a0, fa0, fa1
    (flt.d, (a0, fa0, fa1), "5315b5a2"),  # flt.d a0, fa0, fa1
    (fle.d, (a0, fa0, fa1), "5305b5a2"),  # fle.d a0, fa0, fa1
    (fclass.s, (a0, fa0), "531505e0"),  # fclass.s a0, fa0
    (fclass.d, (a0, fa0), "531505e2"),  # fclass.d a0, fa0
    (fcvt.w.s, (a0, fa0), "537505c0"),  # fcvt.w.s a0, fa0
    (fcvt.wu.s, (a0, fa0), "537515c0"),  # fcvt.wu.s a0, fa0
    (fcvt.l.s, (a0, fa0), "537525c0"),  # fcvt.l.s a0, fa0
    (fcvt.lu.s, (a0, fa0), "537535c0"),  # fcvt.lu.s a0, fa0
    (fcvt.s.w, (fa0, a0), "537505d0"),  # fcvt.s.w fa0, a0
    (fcvt.s.wu, (fa0, a0), "537515d0"),  # fcvt.s.wu fa0, a0
    (fcvt.s.l, (fa0, a0), "537525d0"),  # fcvt.s.l fa0, a0
    (fcvt.s.lu, (fa0, a0), "537535d0"),  # fcvt.s.lu fa0, a0
    (fcvt.w.d, (a0, fa0), "537505c2"),  # fcvt.w.d a0, fa0
    (fcvt.wu.d, (a0, fa0), "537515c2"),  # fcvt.wu.d a0, fa0
    (fcvt.l.d, (a0, fa0), "537525c2"),  # fcvt.l.d a0, fa0
    (fcvt.lu.d, (a0, fa0), "537535c2"),  # fcvt.lu.d a0, fa0
    (fcvt.d.w, (fa0, a0), "530505d2"),  # fcvt.d.w fa0, a0
    (fcvt.d.wu, (fa0, a0), "530515d2"),  # fcvt.d.wu fa0, a0
    (fcvt.d.l, (fa0, a0), "537525d2"),  # fcvt.d.l fa0, a0
    (fcvt.d.lu, (fa0, a0), "537535d2"),  # fcvt.d.lu fa0, a0
    (fcvt.s.d, (fa0, fa1), "53f51540"),  # fcvt.s.d fa0, fa1
    (fcvt.d.s, (fa0, fa1), "53850542"),  # fcvt.d.s fa0, fa1
    (fmv.x.w, (a0, fa0), "530505e0"),  # fmv.x.w a0, fa0
    (fmv.w.x, (fa0, a0), "530505f0"),  # fmv.w.x fa0, a0
    (fmv.x.d, (a0, fa0), "530505e2"),  # fmv.x.d a0, fa0
    (fmv.d.x, (fa0, a0), "530505f2"),  # fmv.d.x fa0, a0
    (sh1add, (a0, a1, a2), "33a5c520"),  # sh1add a0, a1, a2
    (sh2add, (a0, a1, a2), "33c5c520"),  # sh2add a0, a1, a2
    (sh3add, (a0, a1, a2), "33e5c520"),  # sh3add a0, a1, a2
    (add.uw, (a0, a1, a2), "3b85c508"),  # add.uw a0, a1, a2
    (sh1add.uw, (a0, a1, a2), "3ba5c520"),  # sh1add.uw a0, a1, a2
    (sh2add.uw, (a0, a1, a2), "3bc5c520"),  # sh2add.uw a0, a1, a2
    (sh3add.uw, (a0, a1, a2), "3be5c520"),  # sh3add.uw a0, a1, a2
    (slli.uw, (a0, a1, 63), "1b95f50b"),  # slli.uw a0, a1, 63
    (andn, (a0, a1, a2), "33f5c540"),  # andn a0, a1, a2
    (orn, (a0, a1, a2), "33e5c540"),  # orn a0, a1, a2
    (xnor, (a0, a1, a2), "33c5c540"),  # xnor a0, a1, a2
    (clz, (a0, a1), "13950560"),  # clz a0, a1
    (ctz, (a0, a1), "13951560"),  # ctz a0, a1
    (cpop, (a0, a1), "13952560"),  # cpop a0, a1
    (clzw, (a0, a1), "1b950560"),  # clzw a0, a1
    (ctzw, (a0, a1), "1b951560"),  # ctzw a0, a1
    (cpopw, (a0, a1), "1b952560"),  # cpopw a0, a1
    (max_, (a0, a1, a2), "33e5c50a"),  # max a0, a1, a2
    (maxu, (a0, a1, a2), "33f5c50a"),  # maxu a0, a1, a2
    (min_, (a0, a1, a2), "33c5c50a"),  # min a0, a1, a2
    (minu, (a0, a1, a2), "33d5c50a"),  # minu a0, a1, a2
    (sext.b, (a0, a1), "13954560"),  # sext.b a0, a1
    (sext.h, (a0, a1), "13955560"),  # sext.h a0, a1
    (zext.h, (a0, a1), "3bc50508"),  # zext.h a0, a1
    (rol, (a0, a1, a2), "3395c560"),  # rol a0, a1, a2
    (rolw, (a0, a1, a2), "3b95c560"),  # rolw a0, a1, a2
    (ror, (a0, a1, a2), "33d5c560"),  # ror a0, a1, a2
    (rorw, (a0, a1, a2), "3bd5c560"),  # rorw a0, a1, a2
    (rori, (a0, a1, 63), "13d5f563"),  # rori a0, a1, 63
    (roriw, (a0, a1, 31), "1bd5f561"),  # roriw a0, a1, 31
    (orc.b, (a0, a1), "13d57528"),  # orc.b a0, a1
    (rev8, (a0, a1), "13d5856b"),  # rev8 a0, a1
    (nop, (), "13000000"),  # nop
    (mv, (a0, a1), "13850500"),  # mv a0, a1
    (not_, (a0, a1), "13c5f5ff"),  # not a0, a1
    (neg, (a0, a1), "3305b040"),  # neg a0, a1
    (negw, (a0, a1), "3b05b040"),  # negw a0, a1
    (sext.w, (a0, a1), "1b850500"),  # sext.w a0, a1
    (zext.b, (a0, a1), "13f5f50f"),  # zext.b a0, a1
    (zext.w, (a0, a1), "3b850508"),  # zext.w a0, a1
    (seqz, (a0, a1), "13b51500"),  # seqz a0, a1
    (snez, (a0, a1), "3335b000"),  # snez a0, a1
    (sltz, (a0, a1), "33a50500"),  # sltz a0, a1
    (sgtz, (a0, a1), "3325b000"),  # sgtz a0, a1
    (jr, (t0,), "67800200"),  # jr t0
    (ret, (), "67800000"),  # ret
    (fmv.s, (fa0, fa1), "5385b520"),  # fmv.s fa0, fa1
    (fneg.s, (fa0, fa1), "5395b520"),  # fneg.s fa0, fa1
    (fabs.s, (fa0, fa1), "53a5b520"),  # fabs.s fa0, fa1
    (fmv.d, (fa0, fa1), "5385b522"),  # fmv.d fa0, fa1
    (fneg.d, (fa0, fa1), "5395b522"),  # fneg.d fa0, fa1
    (fabs.d, (fa0, fa1), "53a5b522"),  # fabs.d fa0, fa1
    (csrr, (a0, "time"), "732510c0"),  # csrr a0, time
    (csrr, (a0, 0x300), "73250030"),  # csrr a0, 768
    (csrw, ("fcsr", a0), "73103500"),  # csrw fcsr, a0
    (csrs, (0x300, a0), "73200530"),  # csrs 768, a0
    (csrc, ("fflags", a0), "73301500"),  # csrc fflags, a0
    (csrwi, ("fcsr", 5), "73d03200"),  # csrwi fcsr, 5
    (csrsi, (0x300, 31), "73e00f30"),  # csrsi 768, 31
    (csrci, ("fflags", 1), "73f01000"),  # csrci fflags, 1
    (frcsr, (a0,), "73253000"),  # frcsr a0
    (fscsr, (a1,), "73903500"),  # fscsr a1
    (fscsr, (a0, a1), "73953500"),  # fscsr a0, a1
    (frrm, (a0,), "73252000"),  # frrm a0
    (fsrm, (a1,), "73902500"),  # fsrm a1
    (fsrm, (a0, a1), "73952500"),  # fsrm a0, a1
    (fsrmi, (3,), "73d02100"),  # fsrmi 3
    (fsrmi, (a0, 4), "73552200"),  # fsrmi a0, 4
    (frflags, (a0,), "73251000"),  # frflags a0
    (fsflags, (a1,), "73901500"),  # fsflags a1
    (fsflags, (a0, a1), "73951500"),  # fsflags a0, a1
    (fsflagsi, (31,), "73d01f00"),  # fsflagsi 31
    (fsflagsi, (a0, 0), "73551000"),  # fsflagsi a0, 0
    (rdcycle, (a0,), "732500c0"),  # rdcycle a0
    (rdtime, (a0,), "732510c0"),  # rdtime a0
    (rdinstret, (a0,), "732520c0"),  # rdinstret a0
    (sgt, (a0, a1, a2), "3325b600"),  # sgt a0, a1, a2
    (sgtu, (a0, a1, a2), "3335b600"),  # sgtu a0, a1, a2
    (fgt.s, (a0, fa1, fa2), "5315b6a0"),  # fgt.s a0, fa1, fa2
    (fge.s, (a0, fa1, fa2), "5305b6a0"),  # fge.s a0, fa1, fa2
    (fgt.d, (a0, fa1, fa2), "5315b6a2"),  # fgt.d a0, fa1, fa2
    (fge.d, (a0, fa1, fa2), "5305b6a2"),  # fge.d a0, fa1, fa2
    (fmv.x.s, (a0, fa1), "538505e0"),  # fmv.x.s a0, fa1
    (fmv.s.x, (fa0, a1), "538505f0"),  # fmv.s.x fa0, a1
    (jr, (a1, 12), "6780c500"),  # jr a1, 12
    (jalr, (a1, -4), "e780c5ff"),  # jalr a1, -4
    (unimp, (), "731000c0"),  # unimp
    (li, (a0, 0), "13050000"),  # li a0, 0
    (li, (a0, 2047), "1305f07f"),  # li a0, 2047
    (li, (a0, -2048), "13050080"),  # li a0, -2048
    (li, (a0, 2048), "371500001b050580"),  # li a0, 2048
    (li, (a0, -2049), "37f5ffff1b05f57f"),  # li a0, -2049
    (li, (a0, 0x12345678), "375534121b058567"),  # li a0, 305419896
    (li, (a0, 0x7fffffff), "370500801b05f5ff"),  # li a0, 2147483647
    (li, (a0, -0x80000000), "37050080"),  # li a0, -2147483648
    (li, (a0, 0x80000000), "1b0510001315f501"),  # li a0, 2147483648
    (li, (a0, 0xffffffff), "1b051000131505021305f5ff"),  # li a0, 4294967295
    (li, (a0, 0x100000000), "1b05100013150502"),  # li a0, 4294967296
    (li, (t0, 0x123456789abcdef0), "b77224009b82d28a9392e2009382d2c49392c2009382725e9392d200938202ef"),  # li t0, 1311768467463790320
    (li, (a0, 0x7fffffffffffffff), "1b05f0ff1315f5031305f5ff"),  # li a0, 9223372036854775807
    (li, (a0, -0x8000000000000000), "1b05f0ff1315f503"),  # li a0, -9223372036854775808
    (li, (a0, 0xffffffffffffffff), "1305f0ff"),  # li a0, 18446744073709551615
    (lr.w, (a0, mem[a1]), "2fa50510"),  # lr.w a0, (a1)
    (lr.d.aqrl, (t0, mem[sp]), "af320116"),  # lr.d.aqrl t0, (sp)
    (sc.w.rl, (a0, a2, mem[a1]), "2fa5c51a"),  # sc.w.rl a0, a2, (a1)
    (sc.d, (a0, a2, mem[a1]), "2fb5c518"),  # sc.d a0, a2, (a1)
    (amoswap.w.aq, (a0, a2, mem[a1]), "2fa5c50c"),  # amoswap.w.aq a0, a2, (a1)
    (amoadd.d, (a0, a2, mem[a1]), "2fb5c500"),  # amoadd.d a0, a2, (a1)
    (amoxor.w, (a0, a2, mem[a1]), "2fa5c520"),  # amoxor.w a0, a2, (a1)
    (amoand.d.aqrl, (a0, a2, mem[a1]), "2fb5c566"),  # amoand.d.aqrl a0, a2, (a1)
    (amoor.w.rl, (a0, a2, mem[a1]), "2fa5c542"),  # amoor.w.rl a0, a2, (a1)
    (amomin.d, (a0, a2, mem[a1]), "2fb5c580"),  # amomin.d a0, a2, (a1)
    (amomax.w, (a0, a2, mem[a1]), "2fa5c5a0"),  # amomax.w a0, a2, (a1)
    (amominu.d.aq, (a0, a2, mem[a1]), "2fb5c5c4"),  # amominu.d.aq a0, a2, (a1)
    (amomaxu.w, (zero, a2, mem[a1]), "2fa0c5e0"),  # amomaxu.w zero, a2, (a1)
]

# Rounding modes: (fn, ops, rm, bytes).
RM_CASES = [
    (fadd.d, (fa0, fa1, fa2), "rtz", "5395c502"),  # fadd.d fa0, fa1, fa2, rtz
    (fsqrt.s, (fa0, fa1), "rmm", "53c50558"),  # fsqrt.s fa0, fa1, rmm
    (fmadd.d, (fa0, fa1, fa2, fa3), "rdn", "43a5c56a"),  # fmadd.d fa0, fa1, fa2, fa3, rdn
    (fcvt.w.d, (a0, fa0), "rtz", "531505c2"),  # fcvt.w.d a0, fa0, rtz
    (fcvt.l.s, (a0, fa0), "rup", "533525c0"),  # fcvt.l.s a0, fa0, rup
    (fcvt.s.w, (fa0, a0), "rne", "530505d0"),  # fcvt.s.w fa0, a0, rne
    (fcvt.d.l, (fa0, a0), "rne", "530525d2"),  # fcvt.d.l fa0, a0, rne
    (fcvt.s.d, (fa0, fa1), "rne", "53851540"),  # fcvt.s.d fa0, fa1, rne
    (fadd.s, (fa0, fa1, fa2), "dyn", "53f5c500"),  # fadd.s fa0, fa1, fa2, dyn
]

# Label targets: (fn, ops, backward, forward). Backward: the label is bound
# one nop before the instruction (-4). Forward: the label follows the
# instruction and two nops. call and tail are compared with the explicit
# `auipc; jalr %pcrel_lo` pair (GNU as leaves them to the linker).
LABEL_CASES = [
    (beq, (a0, a1, LBL), "e30eb5fe", "6306b500"),  # beq a0, a1, >1
    (bne, (a0, a1, LBL), "e31eb5fe", "6316b500"),  # bne a0, a1, >1
    (blt, (a0, a1, LBL), "e34eb5fe", "6346b500"),  # blt a0, a1, >1
    (bge, (a0, a1, LBL), "e35eb5fe", "6356b500"),  # bge a0, a1, >1
    (bltu, (a0, a1, LBL), "e36eb5fe", "6366b500"),  # bltu a0, a1, >1
    (bgeu, (a0, a1, LBL), "e37eb5fe", "6376b500"),  # bgeu a0, a1, >1
    (beqz, (a0, LBL), "e30e05fe", "63060500"),  # beqz a0, >1
    (bnez, (a0, LBL), "e31e05fe", "63160500"),  # bnez a0, >1
    (blez, (a0, LBL), "e35ea0fe", "6356a000"),  # blez a0, >1
    (bgez, (a0, LBL), "e35e05fe", "63560500"),  # bgez a0, >1
    (bltz, (a0, LBL), "e34e05fe", "63460500"),  # bltz a0, >1
    (bgtz, (a0, LBL), "e34ea0fe", "6346a000"),  # bgtz a0, >1
    (bgt, (a0, a1, LBL), "e3cea5fe", "63c6a500"),  # bgt a0, a1, >1
    (ble, (a0, a1, LBL), "e3dea5fe", "63d6a500"),  # ble a0, a1, >1
    (bgtu, (a0, a1, LBL), "e3eea5fe", "63e6a500"),  # bgtu a0, a1, >1
    (bleu, (a0, a1, LBL), "e3fea5fe", "63f6a500"),  # bleu a0, a1, >1
    (jal, (LBL,), "eff0dfff", "ef00c000"),  # jal >1
    (jal, (a0, LBL), "6ff5dfff", "6f05c000"),  # jal a0, >1
    (j, (LBL,), "6ff0dfff", "6f00c000"),  # j >1
    (lla, (a0, LBL), "170500001305c5ff", "1705000013050501"),  # lla a0, >1
    (la, (t0, LBL), "970200009382c2ff", "9702000093820201"),  # la t0, >1
    (call, (LBL,), "97000000e780c0ff", "97000000e7800001"),  # call >1
    (tail, (LBL,), "170300006700c3ff", "1703000067000301"),  # tail >1
    (lb, (a0, LBL), "170500000305c5ff", "1705000003050501"),  # lb a0, >1
    (lh, (a0, LBL), "170500000315c5ff", "1705000003150501"),  # lh a0, >1
    (lw, (a0, LBL), "170500000325c5ff", "1705000003250501"),  # lw a0, >1
    (ld, (a0, LBL), "170500000335c5ff", "1705000003350501"),  # ld a0, >1
    (lbu, (a0, LBL), "170500000345c5ff", "1705000003450501"),  # lbu a0, >1
    (lhu, (a0, LBL), "170500000355c5ff", "1705000003550501"),  # lhu a0, >1
    (lwu, (s1, LBL), "9704000083e4c4ff", "9704000083e40401"),  # lwu s1, >1
    (sb, (a0, LBL, t0), "97020000238ea2fe", "970200002388a200"),  # sb a0, >1, t0
    (sh, (a0, LBL, t0), "97020000239ea2fe", "970200002398a200"),  # sh a0, >1, t0
    (sw, (a0, LBL, t0), "9702000023aea2fe", "9702000023a8a200"),  # sw a0, >1, t0
    (sd, (a0, LBL, t0), "9702000023bea2fe", "9702000023b8a200"),  # sd a0, >1, t0
    (flw, (fa0, LBL, t0), "9702000007a5c2ff", "9702000007a50201"),  # flw fa0, >1, t0
    (fld, (fa0, LBL, t0), "9702000007b5c2ff", "9702000007b50201"),  # fld fa0, >1, t0
    (fsw, (fa0, LBL, t1), "17030000272ea3fe", "170300002728a300"),  # fsw fa0, >1, t1
    (fsd, (fa0, LBL, t1), "17030000273ea3fe", "170300002738a300"),  # fsd fa0, >1, t1
]
# fmt: on


_TEXT_NAMES = {v: k for k, v in PY_NAMES.items()}
_ATOMIC = ("lr.", "sc.", "amo")


def mnemonic(fn) -> str:
    name = str(fn.__qualname__)
    return _TEXT_NAMES.get(name, name)


def gnu_text(fn, ops, rm=None):
    name = mnemonic(fn)

    def one(o):
        if isinstance(o, MemExpr):
            return f"({o.base})" if name.startswith(_ATOMIC) else f"{o.disp}({o.base})"
        return str(o)

    text = name + (" " + ", ".join(one(o) for o in ops) if ops else "")
    return f"{text}, {rm}" if rm else text


def _case_id(case):
    return gnu_text(case[0], case[1])


def _subst(ops, lbl):
    return [lbl if isinstance(o, str) and o == LBL else o for o in ops]


@pytest.mark.parametrize("fn, ops, want", CASES, ids=[_case_id(c) for c in CASES])
def test_encode(fn, ops, want):
    assert hexof(fn, *ops) == want


@pytest.mark.parametrize("fn, ops, rm, want", RM_CASES, ids=[gnu_text(c[0], c[1], c[2]) for c in RM_CASES])
def test_encode_rounding_mode(fn, ops, rm, want):
    assert hexof(fn, *ops, rm=rm) == want


@pytest.mark.parametrize("fn, ops, back, fwd", LABEL_CASES, ids=[_case_id(c) for c in LABEL_CASES])
def test_label_backward(fn, ops, back, fwd):
    a = Assembler(R)
    lbl = a.label()
    a.nop()
    fn(*_subst(ops, lbl), asm=a)
    assert a.link().data[4:].hex() == back


@pytest.mark.parametrize("fn, ops, back, fwd", LABEL_CASES, ids=[_case_id(c) for c in LABEL_CASES])
def test_label_forward(fn, ops, back, fwd):
    a = Assembler(R)
    lbl = Label()
    fn(*_subst(ops, lbl), asm=a)
    a.nop()
    a.nop()
    a.label(lbl)
    assert a.link().data[:-8].hex() == fwd


# Every lr, sc and AMO mnemonic with its ordering suffix: funct5 from the
# specification, checked by test_atomics and test_oracle_atomics.
ATOMIC_FUNCT5 = {
    "lr": 2, "sc": 3, "amoswap": 1, "amoadd": 0, "amoxor": 4, "amoand": 12, "amoor": 8,
    "amomin": 16, "amomax": 20, "amominu": 24, "amomaxu": 28,
}  # fmt: skip
ATOMICS = [
    (f"{name}.{w}{order}", f5, 2 if w == "w" else 3, bits)
    for name, f5 in ATOMIC_FUNCT5.items()
    for w in "wd"
    for order, bits in (("", 0), (".aq", 2), (".rl", 1), (".aqrl", 3))
]


def _atomic_ops(name):
    return (a0, mem[a1]) if name.startswith("lr.") else (a0, a2, mem[a1])  # noqa: F405


@pytest.mark.parametrize("name, f5, f3, bits", ATOMICS, ids=[a[0] for a in ATOMICS])
def test_atomics(name, f5, f3, bits):
    rs2 = 0 if name.startswith("lr.") else 12
    word = f5 << 27 | bits << 25 | rs2 << 20 | 11 << 15 | f3 << 12 | 10 << 7 | 0x2F
    assert hexof(MNEMONICS[name], *_atomic_ops(name)) == word.to_bytes(4, "little").hex()


def test_every_mnemonic_is_tested():
    tested = {mnemonic(c[0]) for c in CASES + LABEL_CASES} | {a[0] for a in ATOMICS}
    assert set(MNEMONIC_ARGC) - tested == set()
    # Every operand count of every mnemonic, too.
    counts = {(mnemonic(c[0]), len(c[1])) for c in CASES + LABEL_CASES}
    counts |= {(a[0], len(_atomic_ops(a[0]))) for a in ATOMICS}
    assert {(m, n) for m, ns in MNEMONIC_ARGC.items() for n in ns} - counts == set()


def test_default_rounding_modes():
    # Omitted rm is dyn (7), as in GNU as; the exact conversions have rm 0
    # and no keyword.
    assert encode(fadd.d, fa0, fa1, fa2)[0] == encode(fadd.d, fa0, fa1, fa2, rm="dyn")[0]  # noqa: F405
    assert (int.from_bytes(encode(fcvt.w.d, a0, fa0)[0], "little") >> 12) & 7 == 7  # noqa: F405
    assert (int.from_bytes(encode(fcvt.d.w, fa0, a0)[0], "little") >> 12) & 7 == 0  # noqa: F405
    with pytest.raises(TypeError, match="rm"):
        encode(fcvt.d.w, fa0, a0, rm="rtz")  # noqa: F405
    from jita.riscv64.encoder import encode as raw

    with pytest.raises(EncodeError, match="takes no rounding mode"):
        raw(Assembler(R), "fcvt.d.w", (fa0, a0), "rtz")  # noqa: F405
    with pytest.raises(EncodeError, match="unknown rounding mode"):
        encode(fadd.d, fa0, fa1, fa2, rm="up")  # noqa: F405


# -- li ------------------------------------------------------------------------


def _sext(v, bits):
    v &= (1 << bits) - 1
    return v - (1 << bits) if v >> (bits - 1) else v


def simulate_li(words, rd):
    """Run an li expansion: lui, addi, addiw and slli on one register."""
    regs = [0] * 32
    mask = (1 << 64) - 1
    for w in words:
        op, d, f3, s1 = w & 0x7F, (w >> 7) & 31, (w >> 12) & 7, (w >> 15) & 31
        imm = _sext(w >> 20, 12)
        assert d == rd, f"{w:08x} writes x{d}"
        if op == 0x37:
            regs[d] = _sext(w & 0xFFFFF000, 32)
        elif op == 0x13 and f3 == 0:
            regs[d] = _sext(regs[s1] + imm, 64)
        elif op == 0x13 and f3 == 1 and w >> 26 == 0:
            regs[d] = _sext((regs[s1] << ((w >> 20) & 63)) & mask, 64)
        elif op == 0x1B and f3 == 0:
            regs[d] = _sext(regs[s1] + imm, 32)
        else:
            raise AssertionError(f"unexpected word {w:08x} in li")
        regs[0] = 0
    return regs[rd] & mask


def _li_values():
    rng = random.Random(1234)
    vals = {0, 1, -1, 2047, 2048, -2048, -2049, 0xFFF, 0x1000, 0x7FF_FFFF, 0x7FFF_FFFF, -0x8000_0000,
            0x8000_0000, 0xFFFF_FFFF, 0x1_0000_0000, (1 << 63) - 1, -(1 << 63), (1 << 64) - 1}  # fmt: skip
    for bit in range(64):
        for v in (1 << bit, (1 << bit) - 1, -(1 << bit), (1 << bit) + 0x800, (1 << bit) - 0x801):
            vals.add(v)
    for _ in range(3000):
        n = rng.choice((12, 20, 32, 33, 44, 52, 63, 64))
        vals.add(rng.getrandbits(n) - (rng.getrandbits(1) << (n - 1)))
        vals.add(rng.getrandbits(64))
        # Sparse values: a few set bits, the case the shifts are for.
        vals.add(sum(1 << rng.randrange(64) for _ in range(rng.randrange(1, 4))))
    return sorted(v for v in vals if -(1 << 63) <= v < (1 << 64))


LI_VALUES = _li_values()


def test_li_simulated():
    for v in LI_VALUES:
        for rd in (10, 31):
            words = li_words(rd, v)
            assert 1 <= len(words) <= 8, (hex(v), len(words))
            assert simulate_li(words, rd) == v & ((1 << 64) - 1), hex(v)


def test_li_lengths():
    assert len(li_words(10, 5)) == 1
    assert len(li_words(10, 0x12345678)) == 2
    assert len(li_words(10, 0x1000)) == 1  # lui alone
    assert len(li_words(10, 0x123456789ABCDEF0)) == 8
    assert max(len(li_words(10, v)) for v in LI_VALUES) == 8


def test_li_range_and_listing():
    a = Assembler(R)
    a.li(a0, 0x123456789ABCDEF0)  # noqa: F405
    assert a.cur.insns == [(0, 32, "li", (a0, 0x123456789ABCDEF0))]  # noqa: F405
    for bad in (1 << 64, -(1 << 63) - 1):
        with pytest.raises(EncodeError, match="64 bit"):
            a.li(a0, bad)  # noqa: F405
    with pytest.raises(EncodeError, match="expected an immediate"):
        a.li(a0, "x")  # noqa: F405


@requires_riscv64_oracle
def test_li_matches_gnu():
    # One assembly for all values, compared word by word per value.
    text = "\n".join(f"li a0, {v}" for v in LI_VALUES)
    code = assemble(text, arch="riscv64")
    pos = 0
    for v in LI_VALUES:
        want = b"".join(w.to_bytes(4, "little") for w in li_words(10, v))
        assert code[pos : pos + len(want)] == want, hex(v)
        pos += len(want)
    assert pos == len(code)


# -- labels, relocations, externs ------------------------------------------------


def test_string_labels_and_methods():
    a = Assembler(R)
    a.label("top")
    a.addi(a0, a0, -1)  # noqa: F405
    a.bnez(a0, "top")  # noqa: F405
    a.beqz(a0, "out")  # noqa: F405
    a.j("top")
    a.label("out")
    a.ret()
    assert a.link().data.hex() == "1305f5ff" + "e31e05fe" + "63040500" + "6ff05fff" + "67800000"


@pytest.mark.parametrize(
    "fn, ops, limit",
    [
        (beq, (a0, a1), 1 << 12),  # noqa: F405
        (bnez, (a0,), 1 << 12),  # noqa: F405
        (jal, (), 1 << 20),  # noqa: F405
        (j, (), 1 << 20),  # noqa: F405
    ],
)
def test_branch_ranges(fn, ops, limit):
    base = 1 << 32
    a = Assembler(R)
    fn(*ops, Extern("t"), asm=a)
    a.link(base=base, externs={"t": base + limit - 2})
    a.link(base=base, externs={"t": base - limit})
    with pytest.raises(LinkError, match="out of range"):
        a.link(base=base, externs={"t": base + limit})
    with pytest.raises(LinkError, match="out of range"):
        a.link(base=base, externs={"t": base - limit - 2})
    with pytest.raises(LinkError, match="multiple of 2"):
        a.link(base=base, externs={"t": base + 1})


def _pcrel_value(code):
    # auipc hi20 plus the sign extended lo12 of an I-type second word.
    hi = _sext(int.from_bytes(code[:4], "little") & 0xFFFFF000, 32)
    return hi + _sext(int.from_bytes(code[4:8], "little") >> 20, 12)


PCREL_MAX = 0x7FFF_F7FF  # hi20 = 0x7ffff, lo12 = 2047
PCREL_MIN = -0x8000_0800  # hi20 = -0x80000, lo12 = -2048


@pytest.mark.parametrize("fn, ops", [(lla, (a0,)), (call, ()), (ld, (a0,)), (tail, ())])  # noqa: F405
def test_pcrel_ranges(fn, ops):
    base = 1 << 36
    a = Assembler(R)
    fn(*ops, Extern("t"), asm=a)
    for n in (PCREL_MAX, PCREL_MIN, 0x7FF, 0x800, -0x800, -0x801, 0x1234_5678, -0x1234_5678, 0, 1):
        code = a.link(base=base, externs={"t": base + n}).data
        assert _pcrel_value(code) == n
    for n in (PCREL_MAX + 1, PCREL_MIN - 1):
        with pytest.raises(LinkError, match="out of range"):
            a.link(base=base, externs={"t": base + n})


def test_pcrel_store_splits_lo12():
    base = 1 << 36
    a = Assembler(R)
    sd(a0, Extern("t"), t0, asm=a)  # noqa: F405
    for n in (PCREL_MAX, PCREL_MIN, -1, 0x7FF, 0x800):
        code = a.link(base=base, externs={"t": base + n}).data
        w = int.from_bytes(code[4:], "little")
        lo = _sext(((w >> 25) << 5) | ((w >> 7) & 31), 12)
        hi = _sext(int.from_bytes(code[:4], "little") & 0xFFFFF000, 32)
        assert hi + lo == n
        assert w & 0x01FFF07F == 0x00A2B023 & 0x01FFF07F  # sd a0, lo(t0)


def test_split_pcrel():
    assert split_pcrel(0) == (0, 0)
    assert split_pcrel(0x800) == (1, -0x800)
    assert split_pcrel(0x7FF) == (0, 0x7FF)
    assert split_pcrel(-0x801) == (-1, 0x7FF)
    assert split_pcrel(PCREL_MAX) == (0x7FFFF, 0x7FF)
    assert split_pcrel(PCREL_MIN) == (-0x80000, -0x800)
    assert split_pcrel(PCREL_MAX + 1) is None and split_pcrel(PCREL_MIN - 1) is None


def test_patch_kinds_or_into_words():
    buf = bytearray(bytes.fromhex("63000000"))
    B12.apply(buf, 0, 0x1000 - 4, 0x1000)
    assert buf.hex() == "e30e00fe"
    buf = bytearray(bytes.fromhex("6f000000"))
    J20.apply(buf, 0, 0x1000 + 0xFFFFE, 0x1000)
    assert buf.hex() == "6ff0ff7f"
    buf = bytearray(bytes.fromhex("17050000" "13050500"))
    PCREL32.apply(buf, 0, 0x2000 + 0x12345678, 0x2000)
    assert _pcrel_value(buf) == 0x12345678
    assert (B12.size, J20.size, PCREL32.size) == (4, 4, 8)


def test_extern_targets_and_slots():
    base = 0x10000
    a = Assembler(R)
    call(Extern("f"), asm=a)  # noqa: F405
    ld(a0, Extern("g"), asm=a)  # noqa: F405 - loads the 8 bytes at g
    ld(t0, a.extern_slot(Extern("h")), asm=a)  # noqa: F405 - loads h's address
    jalr(t0, asm=a)  # noqa: F405
    img = a.link(base=base, externs={"f": base + 0x800, "g": base - 8, "h": 0x7F00_0000_1234})
    assert _pcrel_value(img.data[0:8]) == 0x800
    assert _pcrel_value(img.data[8:16]) == -8 - 8
    slot = img.section_offsets["externs"]
    assert _pcrel_value(img.data[16:24]) == slot - 16
    assert img.data[slot : slot + 8] == (0x7F00_0000_1234).to_bytes(8, "little")
    with pytest.raises(LinkError, match="out of range"):
        a.link(base=base, externs={"f": base + (1 << 32), "g": 0, "h": 0})


def test_data_directives_with_labels():
    a = Assembler(R)
    lbl = a.label("here")
    a.qword(lbl)
    a.dword(0x12345678)
    assert a.link(base=0x10000).data.hex() == "0000010000000000" + "78563412"


# -- arch object ---------------------------------------------------------------


def test_nop_fill_and_align():
    assert R.NOP == bytes.fromhex("13000000")
    assert R.ARCH.nop_fill(8) == R.NOP * 2
    assert R.ARCH.nop_fill(6) == b"\0\0" + R.NOP
    assert len(R.ARCH.nop_fill(0)) == 0
    a = Assembler(R)
    a.ret()
    a.align(16)
    assert bytes(a.cur.buf) == bytes.fromhex("67800000") + R.NOP * 3


def test_arch_attributes():
    assert R.ARCH.name == "riscv64" and R.ARCH.pointer_size == 8 and R.ARCH.imm_prefix == ""
    assert R.ARCH.insns is INSNS
    assert Assembler(R).arch is R.ARCH and Assembler(R.ARCH).arch is R.ARCH
    assert Assembler("riscv64").arch is R.ARCH


def test_icache_flush(monkeypatch):
    calls = []
    arch = R.Riscv64Arch()
    monkeypatch.setattr(R, "_find_flush", lambda: lambda start, end: calls.append((start, end)))
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")
    arch.icache_flush(0x1000, 64)
    assert calls == []
    monkeypatch.setattr(platform, "machine", lambda: "riscv64")
    arch.icache_flush(0x1000, 64)
    arch.icache_flush(0x2000, 0)
    assert calls == [(0x1000, 0x1040)]


def test_icache_flush_falls_back_to_syscall(monkeypatch):
    import ctypes

    calls = []

    class Syscall:
        def __call__(self, *args):
            calls.append(tuple(getattr(x, "value", x) for x in args))
            return 0

    class FakeLib:
        # No __clear_cache; syscall records its arguments.
        def __init__(self, name, use_errno=False):
            if name is not None:
                raise OSError(name)
            self.syscall = Syscall()

    monkeypatch.setattr(ctypes, "CDLL", FakeLib)
    R._find_flush()(0x1000, 0x1040)
    assert calls == [(259, 0x1000, 0x1040, 0)]


@pytest.mark.parametrize("machine", ["riscv64", "RISCV64"])
def test_host_arch(monkeypatch, machine):
    import jita.core.assembler as asm_mod

    monkeypatch.setattr(asm_mod.platform, "machine", lambda: machine)
    assert Assembler().arch is R.ARCH


def test_star_exports():
    names = set(R.__all__)
    assert {"and_", "or_", "not_", "min_", "max_", "fadd", "fcvt", "lr", "amoadd", "fence", "li", "call",
            "mem", "sp", "zero", "fp", "x0", "f31", "ARCH", "Riscv64Arch", "Riscv64Assembler", "label"} <= names  # fmt: skip
    assert not {"and", "min", "max", "encoder", "ctypes", "platform", "INSNS", "Group"} & names
    assert R.min_ is INSNS["min"] and R.max_ is INSNS["max"]


def test_dotted_mnemonics():
    assert isinstance(fadd, Group) and isinstance(fcvt.w, Group)  # noqa: F405
    assert fadd.d is MNEMONICS["fadd.d"] and fcvt.w.d is MNEMONICS["fcvt.w.d"]  # noqa: F405
    assert fence.i is MNEMONICS["fence.i"] and fence is MNEMONICS["fence"]  # noqa: F405
    assert fmv.d.x is MNEMONICS["fmv.d.x"] and lr.w.aq is MNEMONICS["lr.w.aq"]  # noqa: F405
    assert fadd.d.__name__ == "d" and fadd.d.__qualname__ == "fadd.d"  # noqa: F405
    with pytest.raises(EncodeError, match="fadd is not an instruction, use fadd.d, fadd.s"):
        fadd(fa0, fa1, fa2, asm=Assembler(R))  # noqa: F405
    a = Assembler(R)
    a.fadd.d(fa0, fa1, fa2)  # noqa: F405
    a.fcvt.w.d(a0, fa0, rm="rtz")  # noqa: F405
    a.lr.w.aq(a0, mem[a1])  # noqa: F405
    a.fence.i()
    a.min(a0, a1, a2)  # noqa: F405
    assert bytes(a.cur.buf).hex() == "53f5c502" "531505c2" "2fa50514" "0f100000" "33c5c50a"
    assert all(getattr(f, "__test__", True) is False for f in MNEMONICS.values())


def test_listing():
    from jita.tools.listing import listing

    a = Assembler(R)
    with a:
        label("loop")
        ld(a1, mem[a0 + 8])  # noqa: F405
        slli(a1, a1, 3)  # noqa: F405
        lui(a2, -1)  # noqa: F405
        fcvt.w.d(a0, fa0, rm="rtz")  # noqa: F405
        amoadd.w(a0, a1, mem[a2])  # noqa: F405
        csrrs(a0, 3, zero)  # noqa: F405
        bnez(a0, "loop")  # noqa: F405
        li(t0, 0x12345678)  # noqa: F405
        lla(a0, "loop")  # noqa: F405
    text = listing(a)
    assert "ld a1, 8(a0)" in text
    assert "slli a1, a1, 0x3" in text
    assert "lui a2, 0xfffff" in text
    assert "fcvt.w.d a0, fa0, rtz" in text
    assert "amoadd.w a0, a1, (a2)" in text
    assert "csrrs a0, fcsr, zero" in text
    assert "bnez a0, loop" in text and "b12 -> loop" in text
    assert "li t0, 0x12345678" in text
    assert "lla a0, loop" in text and "pcrel32 -> loop" in text


@requires_riscv64_oracle
def test_listing_matches_objdump():
    # Base instructions print as objdump -M no-aliases prints them, up to
    # the space after each comma.
    from jita.tools.listing import listing

    a = Assembler(R)
    with a:
        addi(a0, sp, -2048)  # noqa: F405
        ld(ra, mem[sp + 2040])  # noqa: F405
        sd(s0, mem[sp - 8])  # noqa: F405
        slli(a0, a1, 63)  # noqa: F405
        sraiw(t0, t1, 7)  # noqa: F405
        lui(a0, 0x12345)  # noqa: F405
        auipc(t6, 0)  # noqa: F405
        csrrw(a0, 0x7C0, a1)  # noqa: F405
        csrrwi(a0, "fflags", 3)  # noqa: F405
        csrrs(a0, "cycle", zero)  # noqa: F405
        fmadd.d(fa0, fa1, fa2, fa3, rm="rne")  # noqa: F405
        fsgnj.s(ft0, ft1, ft2)  # noqa: F405
        lr.d.aqrl(a0, mem[a1])  # noqa: F405
        sc.w(a0, a2, mem[a1])  # noqa: F405
        fence("rw", "w")  # noqa: F405
        fence.i()  # noqa: F405
        sh1add.uw(a0, a1, a2)  # noqa: F405
        rev8(a0, a1)  # noqa: F405
        fld(fs11, mem[t0])  # noqa: F405
    ours = [line.split("  ")[-1].strip().replace(", ", ",") for line in listing(a).splitlines()[1:]]
    theirs = riscv64_disassemble(bytes(a.cur.buf))
    assert ours == theirs


# -- operands ------------------------------------------------------------------


def test_register_objects():
    assert x10 is a0 and f10 is fa0 and fp is s0 and x8 is s0 and x0 is zero  # noqa: F405
    assert gpr(2) is sp and fpr(31) is ft11 and gpr(0) is zero  # noqa: F405
    assert (a0.kind, a0.code, fa0.kind, fa0.code) == ("gp", 10, "fp", 10)  # noqa: F405
    assert str(x10) == "a0" and str(f8) == "fs0"  # noqa: F405
    assert isinstance(a0, X) and isinstance(fa0, F)  # noqa: F405
    with pytest.raises(EncodeError):
        gpr(32)  # noqa: F405
    with pytest.raises(EncodeError):
        fpr(-1)  # noqa: F405


def test_memory_operands():
    assert str(mem[a0]) == "0(a0)"  # noqa: F405
    assert str(mem[sp + 8]) == "8(sp)"  # noqa: F405
    assert str(mem[a0 - 16]) == "-16(a0)"  # noqa: F405
    assert str(mem[8 + sp]) == "8(sp)"  # noqa: F405
    assert mem[sp + 8] == mem[sp + 8] == MemExpr(sp, 8)  # noqa: F405


@pytest.mark.parametrize(
    "build, match",
    [
        (lambda: mem[fa0], "memory base"),  # type: ignore[index]  # noqa: F405  # pyright: ignore[reportArgumentType]
        (lambda: mem[fa0 + 8], "memory base"),  # noqa: F405
        (lambda: mem[a0 + a1], "no index register"),  # type: ignore[operator]  # noqa: F405  # pyright: ignore[reportOperatorIssue]
        (lambda: mem[8], "base register"),  # type: ignore[index]  # noqa: F405  # pyright: ignore[reportArgumentType]
        (lambda: mem[a0, 8], "base register"),  # type: ignore[index]  # noqa: F405  # pyright: ignore[reportArgumentType]
    ],
)
def test_memory_operand_errors(build, match):
    with pytest.raises(EncodeError, match=match):
        build()


@requires_riscv64_oracle
@pytest.mark.parametrize("n", range(32))
def test_every_register_number(n):
    r1, r2, r3 = gpr(n), gpr((n + 7) % 32), gpr((n + 19) % 32)  # noqa: F405
    f1, f2, f3, f4 = fpr(n), fpr((n + 5) % 32), fpr((n + 11) % 32), fpr((n + 29) % 32)  # noqa: F405
    a = Assembler(R)
    with a:
        add(r1, r2, r3)  # noqa: F405
        sd(r1, mem[r2 - 8])  # noqa: F405
        fmadd.d(f1, f2, f3, f4)  # noqa: F405
        fcvt.d.l(f1, r2)  # noqa: F405
    text = f"add {r1}, {r2}, {r3}\nsd {r1}, -8({r2})\nfmadd.d {f1}, {f2}, {f3}, {f4}\nfcvt.d.l {f1}, {r2}"
    assert bytes(a.cur.buf) == assemble(text, arch="riscv64")


# -- encoder errors ------------------------------------------------------------


@pytest.mark.parametrize(
    "fn, ops, match",
    [
        (addi, (a0, a1, 2048), "signed 12 bit"),  # noqa: F405
        (addi, (a0, a1, -2049), "signed 12 bit"),  # noqa: F405
        (addiw, (a0, a1, 1 << 40), "out of range"),  # noqa: F405
        (addi, (a0, fa1, 1), "integer register"),  # noqa: F405
        (add, (a0, a1, 1), "integer register"),  # noqa: F405
        (fadd.d, (fa0, a1, fa2), "floating point register"),  # noqa: F405
        (slli, (a0, a1, 64), "0..63"),  # noqa: F405
        (slli, (a0, a1, -1), "0..63"),  # noqa: F405
        (slliw, (a0, a1, 32), "0..31"),  # noqa: F405
        (rori, (a0, a1, 64), "0..63"),  # noqa: F405
        (lui, (a0, 0x100000), "20 bit"),  # noqa: F405
        (lui, (a0, -0x80001), "20 bit"),  # noqa: F405
        (auipc, (a0, "x"), "expected an immediate"),  # noqa: F405
        (ld, (a0, mem[a1 + 2048]), "signed 12 bit"),  # noqa: F405
        (sd, (a0, mem[a1 - 2049]), "signed 12 bit"),  # noqa: F405
        (ld, (a0, a1 + 8), "mem\\[a1 \\+ 8\\]"),  # noqa: F405
        (ld, (a0, a1), "expected a memory operand"),  # noqa: F405
        (sd, (a0, "x"), "expects 2 or 3 operands|expected a memory operand"),  # noqa: F405
        (lr.w, (a0, mem[a1 + 4]), "no offset"),  # noqa: F405
        (amoadd.d, (a0, a1, a2), "expected a memory operand"),  # noqa: F405
        (csrrw, (a0, 0x1000, a1), "0..0xfff"),  # noqa: F405
        (csrrw, (a0, "mstatus", a1), "unknown CSR name"),  # noqa: F405
        (csrrwi, (a0, "fcsr", 32), "0..31"),  # noqa: F405
        (fence, ("rw",), "expects 0 or 2 operands"),  # noqa: F405
        (fence, ("wr", "rw"), "bad fence set"),  # noqa: F405
        (fence, ("rw", "x"), "bad fence set"),  # noqa: F405
        (fence, ("", "rw"), "empty fence set"),  # noqa: F405
        (fence, (1, 2), "fence set"),  # noqa: F405
        (beq, (a0, a1, 8), "expected a label"),  # noqa: F405
        (j, (a0,), "expected a label"),  # noqa: F405
        (call, (a0,), "expected a label"),  # noqa: F405
        (ret, (a0,), "expects 0 operands"),  # noqa: F405
        (add, (a0, a1), "expects 3 operands"),  # noqa: F405
        (add, (a0, a1, a2, a3), "expects 3 operands"),  # noqa: F405
        (fmv.x.d, (fa0, a0), "integer register"),  # noqa: F405
        (ld, (zero, "x"), "zero cannot hold the auipc address"),  # noqa: F405
        (lw, (zero, Extern("x")), "zero cannot hold the auipc address"),  # noqa: F405
        (sd, (a0, "x", zero), "zero cannot hold the auipc address"),  # noqa: F405
        (fld, (fa0, "x", zero), "zero cannot hold the auipc address"),  # noqa: F405
        (fsw, (fa0, "x", x0), "zero cannot hold the auipc address"),  # noqa: F405
        (la, (zero, "x"), "zero cannot hold the auipc address"),  # noqa: F405
    ],
    ids=lambda v: mnemonic(v) if callable(v) else None,
)
def test_encode_errors(fn, ops, match):
    a = Assembler(R)
    with pytest.raises(EncodeError, match=match):
        fn(*ops, asm=a)
    assert a.pos() == 0 and not a.cur.patches


def test_rejected_instruction_leaves_no_named_label():
    a = Assembler(R)
    with pytest.raises(EncodeError):
        a.sd(fa0, "z", a0)  # noqa: F405
    a.bind(Label("z"))


def test_unknown_instruction():
    from jita.riscv64.encoder import encode as raw

    with pytest.raises(EncodeError, match="unknown instruction"):
        raw(Assembler(R), "frobnicate", ())


def test_lla_zero_is_accepted():
    # GNU as takes `lla zero, lbl` (a no-op pair), unlike `la zero, lbl`.
    a = Assembler(R)
    lla(zero, "x", asm=a)  # noqa: F405
    a.label("x")
    assert a.link().data.hex() == "17000000" "13008000"


@requires_riscv64_oracle
@pytest.mark.parametrize("text", ["ld zero, 1b", "sd a0, 1b, zero", "fld fa0, 1b, zero", "la zero, 1b"])
def test_oracle_rejects_zero_auipc_base(text):
    with pytest.raises(OracleError):
        assemble("1:\n" + text, arch="riscv64")


def test_lui_negative_is_twos_complement():
    # GNU as accepts 0..0xfffff only; jita also takes the negative value
    # of the same field.
    assert hexof(lui, a0, -1) == hexof(lui, a0, 0xFFFFF)  # noqa: F405
    assert hexof(auipc, a0, -0x80000) == hexof(auipc, a0, 0x80000)  # noqa: F405


# -- oracle --------------------------------------------------------------------


@requires_riscv64_oracle
@pytest.mark.parametrize("fn, ops, want", CASES, ids=[_case_id(c) for c in CASES])
def test_oracle(fn, ops, want):
    assert assemble(gnu_text(fn, ops), arch="riscv64").hex() == want


@requires_riscv64_oracle
@pytest.mark.parametrize("fn, ops, rm, want", RM_CASES, ids=[gnu_text(c[0], c[1], c[2]) for c in RM_CASES])
def test_oracle_rounding_mode(fn, ops, rm, want):
    assert assemble(gnu_text(fn, ops, rm), arch="riscv64").hex() == want


@requires_riscv64_oracle
def test_oracle_atomics():
    names = [a[0] for a in ATOMICS]
    text = "\n".join(gnu_text(MNEMONICS[n], _atomic_ops(n)) for n in names)
    want = b"".join(encode(MNEMONICS[n], *_atomic_ops(n))[0] for n in names)
    assert assemble(text, arch="riscv64") == want


_PAIRS = {"call": ("ra", "ra", "ra"), "tail": ("t1", "zero", "t1")}


def _label_text(fn, ops, target):
    name = mnemonic(fn)
    if name in _PAIRS:
        tmp, rd, base = _PAIRS[name]
        return f"2: auipc {tmp}, %pcrel_hi({target})\njalr {rd}, %pcrel_lo(2b)({base})"
    return gnu_text(fn, [target if o == LBL else o for o in ops])


@requires_riscv64_oracle
@pytest.mark.parametrize("fn, ops, back, fwd", LABEL_CASES, ids=[_case_id(c) for c in LABEL_CASES])
def test_oracle_labels(fn, ops, back, fwd):
    # A global label leaves the reference to the linker (see oracle.py).
    code = assemble(".globl T\nT:\nnop\n" + _label_text(fn, ops, "T"), arch="riscv64")
    assert code[4:].hex() == back
    code = assemble(_label_text(fn, ops, "T") + "\nnop\nnop\n.globl T\nT:", arch="riscv64")
    assert code[:-8].hex() == fwd
