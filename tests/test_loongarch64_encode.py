"""loongarch64 encoder tests.

`CASES` and `LABEL_CASES` hold byte-exact expectations produced by GNU as
(loongarch64-linux-gnu-as -mno-relax, linked at address 0) for the
instruction in the comment, so they run without binutils. The oracle
tests check every case against GNU as again when it is installed, and add
checks too broad to list: every register number, random operands for
every template, and `li.w`/`li.d` over many values.
"""

import itertools
import platform
import random
from typing import Any

import pytest
from oracle import OracleError, assemble, loongarch64_disassemble, requires_loongarch64_oracle

import jita.loongarch64 as L
from jita import Assembler, EncodeError, Extern, Label, LinkError
from jita.loongarch64 import *  # noqa: F403
from jita.loongarch64.encoder import MNEMONIC_ARGC, li_words
from jita.loongarch64.insns import INSNS, MNEMONICS, PY_NAMES, Group
from jita.loongarch64.patch import B16, B21, B26, CALL36, PCALA, PCREL20, split_call36, split_pcala
from jita.loongarch64.table import MAP_OP

LBL = "LBL"  # replaced by a Label in LABEL_CASES


def encode(fn, *ops):
    """Encode one instruction into a fresh Assembler, return (bytes, patches)."""
    a = Assembler(L)
    fn(*ops, asm=a)
    return bytes(a.cur.buf), a.cur.patches


def hexof(fn, *ops):
    return encode(fn, *ops)[0].hex()


# fmt: off
CASES = [
    (add.w, (a0, a1, a2), "a4181000"),  # add.w $a0, $a1, $a2
    (add.d, (a0, a1, a2), "a4981000"),  # add.d $a0, $a1, $a2
    (sub.w, (a0, a1, a2), "a4181100"),  # sub.w $a0, $a1, $a2
    (sub.d, (a0, a1, a2), "a4981100"),  # sub.d $a0, $a1, $a2
    (slt, (a0, a1, a2), "a4181200"),  # slt $a0, $a1, $a2
    (sltu, (a0, a1, a2), "a4981200"),  # sltu $a0, $a1, $a2
    (maskeqz, (a0, a1, a2), "a4181300"),  # maskeqz $a0, $a1, $a2
    (masknez, (a0, a1, a2), "a4981300"),  # masknez $a0, $a1, $a2
    (nor, (a0, a1, a2), "a4181400"),  # nor $a0, $a1, $a2
    (and_, (a0, a1, a2), "a4981400"),  # and $a0, $a1, $a2
    (or_, (a0, a1, a2), "a4181500"),  # or $a0, $a1, $a2
    (xor, (a0, a1, a2), "a4981500"),  # xor $a0, $a1, $a2
    (orn, (a0, a1, a2), "a4181600"),  # orn $a0, $a1, $a2
    (andn, (a0, a1, a2), "a4981600"),  # andn $a0, $a1, $a2
    (sll.w, (a0, a1, a2), "a4181700"),  # sll.w $a0, $a1, $a2
    (srl.w, (a0, a1, a2), "a4981700"),  # srl.w $a0, $a1, $a2
    (sra.w, (a0, a1, a2), "a4181800"),  # sra.w $a0, $a1, $a2
    (sll.d, (a0, a1, a2), "a4981800"),  # sll.d $a0, $a1, $a2
    (srl.d, (a0, a1, a2), "a4181900"),  # srl.d $a0, $a1, $a2
    (sra.d, (a0, a1, a2), "a4981900"),  # sra.d $a0, $a1, $a2
    (rotr.w, (a0, a1, a2), "a4181b00"),  # rotr.w $a0, $a1, $a2
    (rotr.d, (a0, a1, a2), "a4981b00"),  # rotr.d $a0, $a1, $a2
    (mul.w, (a0, a1, a2), "a4181c00"),  # mul.w $a0, $a1, $a2
    (mulh.w, (a0, a1, a2), "a4981c00"),  # mulh.w $a0, $a1, $a2
    (mulh.wu, (a0, a1, a2), "a4181d00"),  # mulh.wu $a0, $a1, $a2
    (mul.d, (a0, a1, a2), "a4981d00"),  # mul.d $a0, $a1, $a2
    (mulh.d, (a0, a1, a2), "a4181e00"),  # mulh.d $a0, $a1, $a2
    (mulh.du, (a0, a1, a2), "a4981e00"),  # mulh.du $a0, $a1, $a2
    (mulw.d.w, (a0, a1, a2), "a4181f00"),  # mulw.d.w $a0, $a1, $a2
    (mulw.d.wu, (a0, a1, a2), "a4981f00"),  # mulw.d.wu $a0, $a1, $a2
    (div.w, (a0, a1, a2), "a4182000"),  # div.w $a0, $a1, $a2
    (mod.w, (a0, a1, a2), "a4982000"),  # mod.w $a0, $a1, $a2
    (div.wu, (a0, a1, a2), "a4182100"),  # div.wu $a0, $a1, $a2
    (mod.wu, (a0, a1, a2), "a4982100"),  # mod.wu $a0, $a1, $a2
    (div.d, (a0, a1, a2), "a4182200"),  # div.d $a0, $a1, $a2
    (mod.d, (a0, a1, a2), "a4982200"),  # mod.d $a0, $a1, $a2
    (div.du, (a0, a1, a2), "a4182300"),  # div.du $a0, $a1, $a2
    (mod.du, (a0, a1, a2), "a4982300"),  # mod.du $a0, $a1, $a2
    (crc.w.b.w, (a0, a1, a2), "a4182400"),  # crc.w.b.w $a0, $a1, $a2
    (crc.w.h.w, (a0, a1, a2), "a4982400"),  # crc.w.h.w $a0, $a1, $a2
    (crc.w.w.w, (a0, a1, a2), "a4182500"),  # crc.w.w.w $a0, $a1, $a2
    (crc.w.d.w, (a0, a1, a2), "a4982500"),  # crc.w.d.w $a0, $a1, $a2
    (crcc.w.b.w, (a0, a1, a2), "a4182600"),  # crcc.w.b.w $a0, $a1, $a2
    (crcc.w.h.w, (a0, a1, a2), "a4982600"),  # crcc.w.h.w $a0, $a1, $a2
    (crcc.w.w.w, (a0, a1, a2), "a4182700"),  # crcc.w.w.w $a0, $a1, $a2
    (crcc.w.d.w, (a0, a1, a2), "a4982700"),  # crcc.w.d.w $a0, $a1, $a2
    (alsl.w, (a0, a1, a2, 2), "a4980400"),  # alsl.w $a0, $a1, $a2, 2
    (alsl.w, (t8, s8, ra, 1), "f4070400"),  # alsl.w $t8, $s8, $ra, 1
    (alsl.w, (s0, tp, r21, 4), "57d40500"),  # alsl.w $s0, $tp, $r21, 4
    (alsl.wu, (a0, a1, a2, 2), "a4980600"),  # alsl.wu $a0, $a1, $a2, 2
    (alsl.wu, (t8, s8, ra, 1), "f4070600"),  # alsl.wu $t8, $s8, $ra, 1
    (alsl.wu, (s0, tp, r21, 4), "57d40700"),  # alsl.wu $s0, $tp, $r21, 4
    (alsl.d, (a0, a1, a2, 2), "a4982c00"),  # alsl.d $a0, $a1, $a2, 2
    (alsl.d, (t8, s8, ra, 1), "f4072c00"),  # alsl.d $t8, $s8, $ra, 1
    (alsl.d, (s0, tp, r21, 4), "57d42d00"),  # alsl.d $s0, $tp, $r21, 4
    (bytepick.w, (a0, a1, a2, 1), "a4980800"),  # bytepick.w $a0, $a1, $a2, 1
    (bytepick.w, (t8, s8, ra, 0), "f4070800"),  # bytepick.w $t8, $s8, $ra, 0
    (bytepick.w, (s0, tp, r21, 3), "57d40900"),  # bytepick.w $s0, $tp, $r21, 3
    (bytepick.d, (a0, a1, a2, 5), "a4980e00"),  # bytepick.d $a0, $a1, $a2, 5
    (bytepick.d, (t8, s8, ra, 0), "f4070c00"),  # bytepick.d $t8, $s8, $ra, 0
    (bytepick.d, (s0, tp, r21, 7), "57d40f00"),  # bytepick.d $s0, $tp, $r21, 7
    (slli.w, (a0, a1, 7), "a49c4000"),  # slli.w $a0, $a1, 7
    (slli.w, (t8, s8, 0), "f4834000"),  # slli.w $t8, $s8, 0
    (slli.w, (s0, tp, 31), "57fc4000"),  # slli.w $s0, $tp, 31
    (slli.d, (a0, a1, 33), "a4844100"),  # slli.d $a0, $a1, 33
    (slli.d, (t8, s8, 0), "f4034100"),  # slli.d $t8, $s8, 0
    (slli.d, (s0, tp, 63), "57fc4100"),  # slli.d $s0, $tp, 63
    (srli.w, (a0, a1, 7), "a49c4400"),  # srli.w $a0, $a1, 7
    (srli.w, (t8, s8, 0), "f4834400"),  # srli.w $t8, $s8, 0
    (srli.w, (s0, tp, 31), "57fc4400"),  # srli.w $s0, $tp, 31
    (srli.d, (a0, a1, 33), "a4844500"),  # srli.d $a0, $a1, 33
    (srli.d, (t8, s8, 0), "f4034500"),  # srli.d $t8, $s8, 0
    (srli.d, (s0, tp, 63), "57fc4500"),  # srli.d $s0, $tp, 63
    (srai.w, (a0, a1, 7), "a49c4800"),  # srai.w $a0, $a1, 7
    (srai.w, (t8, s8, 0), "f4834800"),  # srai.w $t8, $s8, 0
    (srai.w, (s0, tp, 31), "57fc4800"),  # srai.w $s0, $tp, 31
    (srai.d, (a0, a1, 33), "a4844900"),  # srai.d $a0, $a1, 33
    (srai.d, (t8, s8, 0), "f4034900"),  # srai.d $t8, $s8, 0
    (srai.d, (s0, tp, 63), "57fc4900"),  # srai.d $s0, $tp, 63
    (rotri.w, (a0, a1, 7), "a49c4c00"),  # rotri.w $a0, $a1, 7
    (rotri.w, (t8, s8, 0), "f4834c00"),  # rotri.w $t8, $s8, 0
    (rotri.w, (s0, tp, 31), "57fc4c00"),  # rotri.w $s0, $tp, 31
    (rotri.d, (a0, a1, 33), "a4844d00"),  # rotri.d $a0, $a1, 33
    (rotri.d, (t8, s8, 0), "f4034d00"),  # rotri.d $t8, $s8, 0
    (rotri.d, (s0, tp, 63), "57fc4d00"),  # rotri.d $s0, $tp, 63
    (bstrins.w, (a0, a1, 15, 3), "a40c6f00"),  # bstrins.w $a0, $a1, 15, 3
    (bstrins.w, (t8, s8, 0, 0), "f4036000"),  # bstrins.w $t8, $s8, 0, 0
    (bstrins.w, (s0, tp, 31, 0), "57007f00"),  # bstrins.w $s0, $tp, 31, 0
    (bstrpick.w, (a0, a1, 15, 3), "a48c6f00"),  # bstrpick.w $a0, $a1, 15, 3
    (bstrpick.w, (t8, s8, 0, 0), "f4836000"),  # bstrpick.w $t8, $s8, 0, 0
    (bstrpick.w, (s0, tp, 31, 0), "57807f00"),  # bstrpick.w $s0, $tp, 31, 0
    (bstrins.d, (a0, a1, 40, 8), "a420a800"),  # bstrins.d $a0, $a1, 40, 8
    (bstrins.d, (t8, s8, 0, 0), "f4038000"),  # bstrins.d $t8, $s8, 0, 0
    (bstrins.d, (s0, tp, 63, 63), "57fcbf00"),  # bstrins.d $s0, $tp, 63, 63
    (bstrpick.d, (a0, a1, 40, 8), "a420e800"),  # bstrpick.d $a0, $a1, 40, 8
    (bstrpick.d, (t8, s8, 0, 0), "f403c000"),  # bstrpick.d $t8, $s8, 0, 0
    (bstrpick.d, (s0, tp, 63, 63), "57fcff00"),  # bstrpick.d $s0, $tp, 63, 63
    (slti, (a0, a1, 8), "a4200002"),  # slti $a0, $a1, 8
    (slti, (t8, s8, -2048), "f4032002"),  # slti $t8, $s8, -2048
    (slti, (s0, tp, 2047), "57fc1f02"),  # slti $s0, $tp, 2047
    (sltui, (a0, a1, 8), "a4204002"),  # sltui $a0, $a1, 8
    (sltui, (t8, s8, -2048), "f4036002"),  # sltui $t8, $s8, -2048
    (sltui, (s0, tp, 2047), "57fc5f02"),  # sltui $s0, $tp, 2047
    (addi.w, (a0, a1, 8), "a4208002"),  # addi.w $a0, $a1, 8
    (addi.w, (t8, s8, -2048), "f403a002"),  # addi.w $t8, $s8, -2048
    (addi.w, (s0, tp, 2047), "57fc9f02"),  # addi.w $s0, $tp, 2047
    (addi.d, (a0, a1, 8), "a420c002"),  # addi.d $a0, $a1, 8
    (addi.d, (t8, s8, -2048), "f403e002"),  # addi.d $t8, $s8, -2048
    (addi.d, (s0, tp, 2047), "57fcdf02"),  # addi.d $s0, $tp, 2047
    (lu52i.d, (a0, a1, 8), "a4200003"),  # lu52i.d $a0, $a1, 8
    (lu52i.d, (t8, s8, -2048), "f4032003"),  # lu52i.d $t8, $s8, -2048
    (lu52i.d, (s0, tp, 2047), "57fc1f03"),  # lu52i.d $s0, $tp, 2047
    (andi, (a0, a1, 0x7ff), "a4fc5f03"),  # andi $a0, $a1, 0x7ff
    (andi, (t8, s8, 0), "f4034003"),  # andi $t8, $s8, 0
    (andi, (s0, tp, 4095), "57fc7f03"),  # andi $s0, $tp, 4095
    (ori, (a0, a1, 0x7ff), "a4fc9f03"),  # ori $a0, $a1, 0x7ff
    (ori, (t8, s8, 0), "f4038003"),  # ori $t8, $s8, 0
    (ori, (s0, tp, 4095), "57fcbf03"),  # ori $s0, $tp, 4095
    (xori, (a0, a1, 0x7ff), "a4fcdf03"),  # xori $a0, $a1, 0x7ff
    (xori, (t8, s8, 0), "f403c003"),  # xori $t8, $s8, 0
    (xori, (s0, tp, 4095), "57fcff03"),  # xori $s0, $tp, 4095
    (addu16i.d, (a0, a1, -1), "a4fcff13"),  # addu16i.d $a0, $a1, -1
    (addu16i.d, (t8, s8, -32768), "f4030012"),  # addu16i.d $t8, $s8, -32768
    (addu16i.d, (s0, tp, 32767), "57fcff11"),  # addu16i.d $s0, $tp, 32767
    (lu12i.w, (a0, 0x12345), "a4682414"),  # lu12i.w $a0, 0x12345
    (lu12i.w, (t8, -0x80000), "14000015"),  # lu12i.w $t8, -0x80000
    (lu12i.w, (s0, 0x7ffff), "f7ffff14"),  # lu12i.w $s0, 0x7ffff
    (lu32i.d, (a0, 0x12345), "a4682416"),  # lu32i.d $a0, 0x12345
    (lu32i.d, (t8, -0x80000), "14000017"),  # lu32i.d $t8, -0x80000
    (lu32i.d, (s0, 0x7ffff), "f7ffff16"),  # lu32i.d $s0, 0x7ffff
    (pcaddi, (a0, 0x12345), "a4682418"),  # pcaddi $a0, 0x12345
    (pcaddi, (t8, -0x80000), "14000019"),  # pcaddi $t8, -0x80000
    (pcaddi, (s0, 0x7ffff), "f7ffff18"),  # pcaddi $s0, 0x7ffff
    (pcalau12i, (a0, 0x12345), "a468241a"),  # pcalau12i $a0, 0x12345
    (pcalau12i, (t8, -0x80000), "1400001b"),  # pcalau12i $t8, -0x80000
    (pcalau12i, (s0, 0x7ffff), "f7ffff1a"),  # pcalau12i $s0, 0x7ffff
    (pcaddu12i, (a0, 0x12345), "a468241c"),  # pcaddu12i $a0, 0x12345
    (pcaddu12i, (t8, -0x80000), "1400001d"),  # pcaddu12i $t8, -0x80000
    (pcaddu12i, (s0, 0x7ffff), "f7ffff1c"),  # pcaddu12i $s0, 0x7ffff
    (pcaddu18i, (a0, 0x12345), "a468241e"),  # pcaddu18i $a0, 0x12345
    (pcaddu18i, (t8, -0x80000), "1400001f"),  # pcaddu18i $t8, -0x80000
    (pcaddu18i, (s0, 0x7ffff), "f7ffff1e"),  # pcaddu18i $s0, 0x7ffff
    (clo.w, (a0, a1), "a4100000"),  # clo.w $a0, $a1
    (clz.w, (a0, a1), "a4140000"),  # clz.w $a0, $a1
    (cto.w, (a0, a1), "a4180000"),  # cto.w $a0, $a1
    (ctz.w, (a0, a1), "a41c0000"),  # ctz.w $a0, $a1
    (clo.d, (a0, a1), "a4200000"),  # clo.d $a0, $a1
    (clz.d, (a0, a1), "a4240000"),  # clz.d $a0, $a1
    (cto.d, (a0, a1), "a4280000"),  # cto.d $a0, $a1
    (ctz.d, (a0, a1), "a42c0000"),  # ctz.d $a0, $a1
    (revb._2h, (a0, a1), "a4300000"),  # revb.2h $a0, $a1
    (revb._4h, (a0, a1), "a4340000"),  # revb.4h $a0, $a1
    (revb._2w, (a0, a1), "a4380000"),  # revb.2w $a0, $a1
    (revb.d, (a0, a1), "a43c0000"),  # revb.d $a0, $a1
    (revh._2w, (a0, a1), "a4400000"),  # revh.2w $a0, $a1
    (revh.d, (a0, a1), "a4440000"),  # revh.d $a0, $a1
    (bitrev._4b, (a0, a1), "a4480000"),  # bitrev.4b $a0, $a1
    (bitrev._8b, (a0, a1), "a44c0000"),  # bitrev.8b $a0, $a1
    (bitrev.w, (a0, a1), "a4500000"),  # bitrev.w $a0, $a1
    (bitrev.d, (a0, a1), "a4540000"),  # bitrev.d $a0, $a1
    (ext.w.h, (a0, a1), "a4580000"),  # ext.w.h $a0, $a1
    (ext.w.b, (a0, a1), "a45c0000"),  # ext.w.b $a0, $a1
    (rdtimel.w, (a0, a1), "a4600000"),  # rdtimel.w $a0, $a1
    (rdtimeh.w, (a0, a1), "a4640000"),  # rdtimeh.w $a0, $a1
    (rdtime.d, (a0, a1), "a4680000"),  # rdtime.d $a0, $a1
    (cpucfg, (a0, a1), "a46c0000"),  # cpucfg $a0, $a1
    (asrtle.d, (a1, a2), "a0180100"),  # asrtle.d $a1, $a2
    (asrtgt.d, (a1, a2), "a0980100"),  # asrtgt.d $a1, $a2
    (break_, (0x7ff,), "ff072a00"),  # break 0x7ff
    (break_, (0,), "00002a00"),  # break 0
    (break_, (0x7fff,), "ff7f2a00"),  # break 0x7fff
    (dbcl, (0x7ff,), "ff872a00"),  # dbcl 0x7ff
    (dbcl, (0,), "00802a00"),  # dbcl 0
    (dbcl, (0x7fff,), "ffff2a00"),  # dbcl 0x7fff
    (syscall, (0x7ff,), "ff072b00"),  # syscall 0x7ff
    (syscall, (0,), "00002b00"),  # syscall 0
    (syscall, (0x7fff,), "ff7f2b00"),  # syscall 0x7fff
    (dbar, (0x7ff,), "ff077238"),  # dbar 0x7ff
    (dbar, (0,), "00007238"),  # dbar 0
    (dbar, (0x7fff,), "ff7f7238"),  # dbar 0x7fff
    (ibar, (0x7ff,), "ff877238"),  # ibar 0x7ff
    (ibar, (0,), "00807238"),  # ibar 0
    (ibar, (0x7fff,), "ffff7238"),  # ibar 0x7fff
    (ld.b, (a0, a1, 8), "a4200028"),  # ld.b $a0, $a1, 8
    (ld.b, (t8, s8, -2048), "f4032028"),  # ld.b $t8, $s8, -2048
    (ld.b, (s0, tp, 2047), "57fc1f28"),  # ld.b $s0, $tp, 2047
    (ld.h, (a0, a1, 8), "a4204028"),  # ld.h $a0, $a1, 8
    (ld.h, (t8, s8, -2048), "f4036028"),  # ld.h $t8, $s8, -2048
    (ld.h, (s0, tp, 2047), "57fc5f28"),  # ld.h $s0, $tp, 2047
    (ld.w, (a0, a1, 8), "a4208028"),  # ld.w $a0, $a1, 8
    (ld.w, (t8, s8, -2048), "f403a028"),  # ld.w $t8, $s8, -2048
    (ld.w, (s0, tp, 2047), "57fc9f28"),  # ld.w $s0, $tp, 2047
    (ld.d, (a0, a1, 8), "a420c028"),  # ld.d $a0, $a1, 8
    (ld.d, (t8, s8, -2048), "f403e028"),  # ld.d $t8, $s8, -2048
    (ld.d, (s0, tp, 2047), "57fcdf28"),  # ld.d $s0, $tp, 2047
    (ld.bu, (a0, a1, 8), "a420002a"),  # ld.bu $a0, $a1, 8
    (ld.bu, (t8, s8, -2048), "f403202a"),  # ld.bu $t8, $s8, -2048
    (ld.bu, (s0, tp, 2047), "57fc1f2a"),  # ld.bu $s0, $tp, 2047
    (ld.hu, (a0, a1, 8), "a420402a"),  # ld.hu $a0, $a1, 8
    (ld.hu, (t8, s8, -2048), "f403602a"),  # ld.hu $t8, $s8, -2048
    (ld.hu, (s0, tp, 2047), "57fc5f2a"),  # ld.hu $s0, $tp, 2047
    (ld.wu, (a0, a1, 8), "a420802a"),  # ld.wu $a0, $a1, 8
    (ld.wu, (t8, s8, -2048), "f403a02a"),  # ld.wu $t8, $s8, -2048
    (ld.wu, (s0, tp, 2047), "57fc9f2a"),  # ld.wu $s0, $tp, 2047
    (st.b, (a0, a1, 8), "a4200029"),  # st.b $a0, $a1, 8
    (st.b, (t8, s8, -2048), "f4032029"),  # st.b $t8, $s8, -2048
    (st.b, (s0, tp, 2047), "57fc1f29"),  # st.b $s0, $tp, 2047
    (st.h, (a0, a1, 8), "a4204029"),  # st.h $a0, $a1, 8
    (st.h, (t8, s8, -2048), "f4036029"),  # st.h $t8, $s8, -2048
    (st.h, (s0, tp, 2047), "57fc5f29"),  # st.h $s0, $tp, 2047
    (st.w, (a0, a1, 8), "a4208029"),  # st.w $a0, $a1, 8
    (st.w, (t8, s8, -2048), "f403a029"),  # st.w $t8, $s8, -2048
    (st.w, (s0, tp, 2047), "57fc9f29"),  # st.w $s0, $tp, 2047
    (st.d, (a0, a1, 8), "a420c029"),  # st.d $a0, $a1, 8
    (st.d, (t8, s8, -2048), "f403e029"),  # st.d $t8, $s8, -2048
    (st.d, (s0, tp, 2047), "57fcdf29"),  # st.d $s0, $tp, 2047
    (preld, (3, a1, 8), "a320c02a"),  # preld 3, $a1, 8
    (preld, (0, s8, -2048), "e003e02a"),  # preld 0, $s8, -2048
    (preld, (31, tp, 2047), "5ffcdf2a"),  # preld 31, $tp, 2047
    (ldx.b, (a0, a1, a2), "a4180038"),  # ldx.b $a0, $a1, $a2
    (ldx.h, (a0, a1, a2), "a4180438"),  # ldx.h $a0, $a1, $a2
    (ldx.w, (a0, a1, a2), "a4180838"),  # ldx.w $a0, $a1, $a2
    (ldx.d, (a0, a1, a2), "a4180c38"),  # ldx.d $a0, $a1, $a2
    (stx.b, (a0, a1, a2), "a4181038"),  # stx.b $a0, $a1, $a2
    (stx.h, (a0, a1, a2), "a4181438"),  # stx.h $a0, $a1, $a2
    (stx.w, (a0, a1, a2), "a4181838"),  # stx.w $a0, $a1, $a2
    (stx.d, (a0, a1, a2), "a4181c38"),  # stx.d $a0, $a1, $a2
    (ldx.bu, (a0, a1, a2), "a4182038"),  # ldx.bu $a0, $a1, $a2
    (ldx.hu, (a0, a1, a2), "a4182438"),  # ldx.hu $a0, $a1, $a2
    (ldx.wu, (a0, a1, a2), "a4182838"),  # ldx.wu $a0, $a1, $a2
    (preldx, (3, a1, a2), "a3182c38"),  # preldx 3, $a1, $a2
    (preldx, (0, s8, ra), "e0072c38"),  # preldx 0, $s8, $ra
    (preldx, (31, tp, r21), "5f542c38"),  # preldx 31, $tp, $r21
    (ldptr.w, (a0, a1, 16), "a4100024"),  # ldptr.w $a0, $a1, 16
    (ldptr.w, (t8, s8, -32768), "f4038024"),  # ldptr.w $t8, $s8, -32768
    (ldptr.w, (s0, tp, 32764), "57fc7f24"),  # ldptr.w $s0, $tp, 32764
    (stptr.w, (a0, a1, 16), "a4100025"),  # stptr.w $a0, $a1, 16
    (stptr.w, (t8, s8, -32768), "f4038025"),  # stptr.w $t8, $s8, -32768
    (stptr.w, (s0, tp, 32764), "57fc7f25"),  # stptr.w $s0, $tp, 32764
    (ldptr.d, (a0, a1, 16), "a4100026"),  # ldptr.d $a0, $a1, 16
    (ldptr.d, (t8, s8, -32768), "f4038026"),  # ldptr.d $t8, $s8, -32768
    (ldptr.d, (s0, tp, 32764), "57fc7f26"),  # ldptr.d $s0, $tp, 32764
    (stptr.d, (a0, a1, 16), "a4100027"),  # stptr.d $a0, $a1, 16
    (stptr.d, (t8, s8, -32768), "f4038027"),  # stptr.d $t8, $s8, -32768
    (stptr.d, (s0, tp, 32764), "57fc7f27"),  # stptr.d $s0, $tp, 32764
    (ldgt.b, (a0, a1, a2), "a4187838"),  # ldgt.b $a0, $a1, $a2
    (ldgt.h, (a0, a1, a2), "a4987838"),  # ldgt.h $a0, $a1, $a2
    (ldgt.w, (a0, a1, a2), "a4187938"),  # ldgt.w $a0, $a1, $a2
    (ldgt.d, (a0, a1, a2), "a4987938"),  # ldgt.d $a0, $a1, $a2
    (ldle.b, (a0, a1, a2), "a4187a38"),  # ldle.b $a0, $a1, $a2
    (ldle.h, (a0, a1, a2), "a4987a38"),  # ldle.h $a0, $a1, $a2
    (ldle.w, (a0, a1, a2), "a4187b38"),  # ldle.w $a0, $a1, $a2
    (ldle.d, (a0, a1, a2), "a4987b38"),  # ldle.d $a0, $a1, $a2
    (stgt.b, (a0, a1, a2), "a4187c38"),  # stgt.b $a0, $a1, $a2
    (stgt.h, (a0, a1, a2), "a4987c38"),  # stgt.h $a0, $a1, $a2
    (stgt.w, (a0, a1, a2), "a4187d38"),  # stgt.w $a0, $a1, $a2
    (stgt.d, (a0, a1, a2), "a4987d38"),  # stgt.d $a0, $a1, $a2
    (stle.b, (a0, a1, a2), "a4187e38"),  # stle.b $a0, $a1, $a2
    (stle.h, (a0, a1, a2), "a4987e38"),  # stle.h $a0, $a1, $a2
    (stle.w, (a0, a1, a2), "a4187f38"),  # stle.w $a0, $a1, $a2
    (stle.d, (a0, a1, a2), "a4987f38"),  # stle.d $a0, $a1, $a2
    (ll.w, (a0, a1, 16), "a4100020"),  # ll.w $a0, $a1, 16
    (ll.w, (t8, s8, -32768), "f4038020"),  # ll.w $t8, $s8, -32768
    (ll.w, (s0, tp, 32764), "57fc7f20"),  # ll.w $s0, $tp, 32764
    (sc.w, (a0, a1, 16), "a4100021"),  # sc.w $a0, $a1, 16
    (sc.w, (t8, s8, -32768), "f4038021"),  # sc.w $t8, $s8, -32768
    (sc.w, (s0, tp, 32764), "57fc7f21"),  # sc.w $s0, $tp, 32764
    (ll.d, (a0, a1, 16), "a4100022"),  # ll.d $a0, $a1, 16
    (ll.d, (t8, s8, -32768), "f4038022"),  # ll.d $t8, $s8, -32768
    (ll.d, (s0, tp, 32764), "57fc7f22"),  # ll.d $s0, $tp, 32764
    (sc.d, (a0, a1, 16), "a4100023"),  # sc.d $a0, $a1, 16
    (sc.d, (t8, s8, -32768), "f4038023"),  # sc.d $t8, $s8, -32768
    (sc.d, (s0, tp, 32764), "57fc7f23"),  # sc.d $s0, $tp, 32764
    (jirl, (a0, a1, 8), "a408004c"),  # jirl $a0, $a1, 8
    (jirl, (t8, s8, -131072), "f403004e"),  # jirl $t8, $s8, -131072
    (jirl, (s0, tp, 131068), "57fcff4d"),  # jirl $s0, $tp, 131068
    (fadd.s, (fa0, fa1, fa2), "20880001"),  # fadd.s $fa0, $fa1, $fa2
    (fsub.s, (fa0, fa1, fa2), "20880201"),  # fsub.s $fa0, $fa1, $fa2
    (fmul.s, (fa0, fa1, fa2), "20880401"),  # fmul.s $fa0, $fa1, $fa2
    (fdiv.s, (fa0, fa1, fa2), "20880601"),  # fdiv.s $fa0, $fa1, $fa2
    (fmax.s, (fa0, fa1, fa2), "20880801"),  # fmax.s $fa0, $fa1, $fa2
    (fmin.s, (fa0, fa1, fa2), "20880a01"),  # fmin.s $fa0, $fa1, $fa2
    (fmaxa.s, (fa0, fa1, fa2), "20880c01"),  # fmaxa.s $fa0, $fa1, $fa2
    (fmina.s, (fa0, fa1, fa2), "20880e01"),  # fmina.s $fa0, $fa1, $fa2
    (fscaleb.s, (fa0, fa1, fa2), "20881001"),  # fscaleb.s $fa0, $fa1, $fa2
    (fcopysign.s, (fa0, fa1, fa2), "20881201"),  # fcopysign.s $fa0, $fa1, $fa2
    (fadd.d, (fa0, fa1, fa2), "20080101"),  # fadd.d $fa0, $fa1, $fa2
    (fsub.d, (fa0, fa1, fa2), "20080301"),  # fsub.d $fa0, $fa1, $fa2
    (fmul.d, (fa0, fa1, fa2), "20080501"),  # fmul.d $fa0, $fa1, $fa2
    (fdiv.d, (fa0, fa1, fa2), "20080701"),  # fdiv.d $fa0, $fa1, $fa2
    (fmax.d, (fa0, fa1, fa2), "20080901"),  # fmax.d $fa0, $fa1, $fa2
    (fmin.d, (fa0, fa1, fa2), "20080b01"),  # fmin.d $fa0, $fa1, $fa2
    (fmaxa.d, (fa0, fa1, fa2), "20080d01"),  # fmaxa.d $fa0, $fa1, $fa2
    (fmina.d, (fa0, fa1, fa2), "20080f01"),  # fmina.d $fa0, $fa1, $fa2
    (fscaleb.d, (fa0, fa1, fa2), "20081101"),  # fscaleb.d $fa0, $fa1, $fa2
    (fcopysign.d, (fa0, fa1, fa2), "20081301"),  # fcopysign.d $fa0, $fa1, $fa2
    (fabs.s, (fa0, fa1), "20041401"),  # fabs.s $fa0, $fa1
    (fneg.s, (fa0, fa1), "20141401"),  # fneg.s $fa0, $fa1
    (flogb.s, (fa0, fa1), "20241401"),  # flogb.s $fa0, $fa1
    (fclass.s, (fa0, fa1), "20341401"),  # fclass.s $fa0, $fa1
    (fsqrt.s, (fa0, fa1), "20441401"),  # fsqrt.s $fa0, $fa1
    (frecip.s, (fa0, fa1), "20541401"),  # frecip.s $fa0, $fa1
    (frsqrt.s, (fa0, fa1), "20641401"),  # frsqrt.s $fa0, $fa1
    (fmov.s, (fa0, fa1), "20941401"),  # fmov.s $fa0, $fa1
    (fabs.d, (fa0, fa1), "20081401"),  # fabs.d $fa0, $fa1
    (fneg.d, (fa0, fa1), "20181401"),  # fneg.d $fa0, $fa1
    (flogb.d, (fa0, fa1), "20281401"),  # flogb.d $fa0, $fa1
    (fclass.d, (fa0, fa1), "20381401"),  # fclass.d $fa0, $fa1
    (fsqrt.d, (fa0, fa1), "20481401"),  # fsqrt.d $fa0, $fa1
    (frecip.d, (fa0, fa1), "20581401"),  # frecip.d $fa0, $fa1
    (frsqrt.d, (fa0, fa1), "20681401"),  # frsqrt.d $fa0, $fa1
    (fmov.d, (fa0, fa1), "20981401"),  # fmov.d $fa0, $fa1
    (fmadd.s, (fa0, fa1, fa2, fa3), "20881108"),  # fmadd.s $fa0, $fa1, $fa2, $fa3
    (fmsub.s, (fa0, fa1, fa2, fa3), "20885108"),  # fmsub.s $fa0, $fa1, $fa2, $fa3
    (fnmadd.s, (fa0, fa1, fa2, fa3), "20889108"),  # fnmadd.s $fa0, $fa1, $fa2, $fa3
    (fnmsub.s, (fa0, fa1, fa2, fa3), "2088d108"),  # fnmsub.s $fa0, $fa1, $fa2, $fa3
    (fmadd.d, (fa0, fa1, fa2, fa3), "20882108"),  # fmadd.d $fa0, $fa1, $fa2, $fa3
    (fmsub.d, (fa0, fa1, fa2, fa3), "20886108"),  # fmsub.d $fa0, $fa1, $fa2, $fa3
    (fnmadd.d, (fa0, fa1, fa2, fa3), "2088a108"),  # fnmadd.d $fa0, $fa1, $fa2, $fa3
    (fnmsub.d, (fa0, fa1, fa2, fa3), "2088e108"),  # fnmsub.d $fa0, $fa1, $fa2, $fa3
    (fsel, (fa0, fa1, fa2, fcc3), "2088010d"),  # fsel $fa0, $fa1, $fa2, $fcc3
    (fsel, (ft15, fs7, fa7, fcc0), "f71f000d"),  # fsel $ft15, $fs7, $fa7, $fcc0
    (fsel, (fs0, ft8, fs3, fcc7), "18ee030d"),  # fsel $fs0, $ft8, $fs3, $fcc7
    (movgr2fr.w, (fa0, a1), "a0a41401"),  # movgr2fr.w $fa0, $a1
    (movgr2fr.d, (fa0, a1), "a0a81401"),  # movgr2fr.d $fa0, $a1
    (movgr2frh.w, (fa0, a1), "a0ac1401"),  # movgr2frh.w $fa0, $a1
    (movfr2gr.s, (a0, fa1), "24b41401"),  # movfr2gr.s $a0, $fa1
    (movfr2gr.d, (a0, fa1), "24b81401"),  # movfr2gr.d $a0, $fa1
    (movfrh2gr.s, (a0, fa1), "24bc1401"),  # movfrh2gr.s $a0, $fa1
    (movgr2fcsr, (fcsr1, a1), "a1c01401"),  # movgr2fcsr $fcsr1, $a1
    (movgr2fcsr, (fcsr0, s8), "e0c31401"),  # movgr2fcsr $fcsr0, $s8
    (movgr2fcsr, (fcsr3, tp), "43c01401"),  # movgr2fcsr $fcsr3, $tp
    (movfcsr2gr, (a0, fcsr2), "44c81401"),  # movfcsr2gr $a0, $fcsr2
    (movfcsr2gr, (t8, fcsr0), "14c81401"),  # movfcsr2gr $t8, $fcsr0
    (movfcsr2gr, (s0, fcsr3), "77c81401"),  # movfcsr2gr $s0, $fcsr3
    (movfr2cf, (fcc1, fa1), "21d01401"),  # movfr2cf $fcc1, $fa1
    (movfr2cf, (fcc0, fs7), "e0d31401"),  # movfr2cf $fcc0, $fs7
    (movfr2cf, (fcc7, ft8), "07d21401"),  # movfr2cf $fcc7, $ft8
    (movcf2fr, (fa0, fcc2), "40d41401"),  # movcf2fr $fa0, $fcc2
    (movcf2fr, (ft15, fcc0), "17d41401"),  # movcf2fr $ft15, $fcc0
    (movcf2fr, (fs0, fcc7), "f8d41401"),  # movcf2fr $fs0, $fcc7
    (movgr2cf, (fcc1, a1), "a1d81401"),  # movgr2cf $fcc1, $a1
    (movgr2cf, (fcc0, s8), "e0db1401"),  # movgr2cf $fcc0, $s8
    (movgr2cf, (fcc7, tp), "47d81401"),  # movgr2cf $fcc7, $tp
    (movcf2gr, (a0, fcc2), "44dc1401"),  # movcf2gr $a0, $fcc2
    (movcf2gr, (t8, fcc0), "14dc1401"),  # movcf2gr $t8, $fcc0
    (movcf2gr, (s0, fcc7), "f7dc1401"),  # movcf2gr $s0, $fcc7
    (fcvt.s.d, (fa0, fa1), "20181901"),  # fcvt.s.d $fa0, $fa1
    (fcvt.d.s, (fa0, fa1), "20241901"),  # fcvt.d.s $fa0, $fa1
    (ftintrm.w.s, (fa0, fa1), "20041a01"),  # ftintrm.w.s $fa0, $fa1
    (ftintrm.w.d, (fa0, fa1), "20081a01"),  # ftintrm.w.d $fa0, $fa1
    (ftintrm.l.s, (fa0, fa1), "20241a01"),  # ftintrm.l.s $fa0, $fa1
    (ftintrm.l.d, (fa0, fa1), "20281a01"),  # ftintrm.l.d $fa0, $fa1
    (ftintrp.w.s, (fa0, fa1), "20441a01"),  # ftintrp.w.s $fa0, $fa1
    (ftintrp.w.d, (fa0, fa1), "20481a01"),  # ftintrp.w.d $fa0, $fa1
    (ftintrp.l.s, (fa0, fa1), "20641a01"),  # ftintrp.l.s $fa0, $fa1
    (ftintrp.l.d, (fa0, fa1), "20681a01"),  # ftintrp.l.d $fa0, $fa1
    (ftintrz.w.s, (fa0, fa1), "20841a01"),  # ftintrz.w.s $fa0, $fa1
    (ftintrz.w.d, (fa0, fa1), "20881a01"),  # ftintrz.w.d $fa0, $fa1
    (ftintrz.l.s, (fa0, fa1), "20a41a01"),  # ftintrz.l.s $fa0, $fa1
    (ftintrz.l.d, (fa0, fa1), "20a81a01"),  # ftintrz.l.d $fa0, $fa1
    (ftintrne.w.s, (fa0, fa1), "20c41a01"),  # ftintrne.w.s $fa0, $fa1
    (ftintrne.w.d, (fa0, fa1), "20c81a01"),  # ftintrne.w.d $fa0, $fa1
    (ftintrne.l.s, (fa0, fa1), "20e41a01"),  # ftintrne.l.s $fa0, $fa1
    (ftintrne.l.d, (fa0, fa1), "20e81a01"),  # ftintrne.l.d $fa0, $fa1
    (ftint.w.s, (fa0, fa1), "20041b01"),  # ftint.w.s $fa0, $fa1
    (ftint.w.d, (fa0, fa1), "20081b01"),  # ftint.w.d $fa0, $fa1
    (ftint.l.s, (fa0, fa1), "20241b01"),  # ftint.l.s $fa0, $fa1
    (ftint.l.d, (fa0, fa1), "20281b01"),  # ftint.l.d $fa0, $fa1
    (ffint.s.w, (fa0, fa1), "20101d01"),  # ffint.s.w $fa0, $fa1
    (ffint.s.l, (fa0, fa1), "20181d01"),  # ffint.s.l $fa0, $fa1
    (ffint.d.w, (fa0, fa1), "20201d01"),  # ffint.d.w $fa0, $fa1
    (ffint.d.l, (fa0, fa1), "20281d01"),  # ffint.d.l $fa0, $fa1
    (frint.s, (fa0, fa1), "20441e01"),  # frint.s $fa0, $fa1
    (frint.d, (fa0, fa1), "20481e01"),  # frint.d $fa0, $fa1
    (fld.s, (fa0, a1, 8), "a020002b"),  # fld.s $fa0, $a1, 8
    (fld.s, (ft15, s8, -2048), "f703202b"),  # fld.s $ft15, $s8, -2048
    (fld.s, (fs0, tp, 2047), "58fc1f2b"),  # fld.s $fs0, $tp, 2047
    (fst.s, (fa0, a1, 8), "a020402b"),  # fst.s $fa0, $a1, 8
    (fst.s, (ft15, s8, -2048), "f703602b"),  # fst.s $ft15, $s8, -2048
    (fst.s, (fs0, tp, 2047), "58fc5f2b"),  # fst.s $fs0, $tp, 2047
    (fld.d, (fa0, a1, 8), "a020802b"),  # fld.d $fa0, $a1, 8
    (fld.d, (ft15, s8, -2048), "f703a02b"),  # fld.d $ft15, $s8, -2048
    (fld.d, (fs0, tp, 2047), "58fc9f2b"),  # fld.d $fs0, $tp, 2047
    (fst.d, (fa0, a1, 8), "a020c02b"),  # fst.d $fa0, $a1, 8
    (fst.d, (ft15, s8, -2048), "f703e02b"),  # fst.d $ft15, $s8, -2048
    (fst.d, (fs0, tp, 2047), "58fcdf2b"),  # fst.d $fs0, $tp, 2047
    (fldx.s, (fa0, a1, a2), "a0183038"),  # fldx.s $fa0, $a1, $a2
    (fldx.d, (fa0, a1, a2), "a0183438"),  # fldx.d $fa0, $a1, $a2
    (fstx.s, (fa0, a1, a2), "a0183838"),  # fstx.s $fa0, $a1, $a2
    (fstx.d, (fa0, a1, a2), "a0183c38"),  # fstx.d $fa0, $a1, $a2
    (fldgt.s, (fa0, a1, a2), "a0187438"),  # fldgt.s $fa0, $a1, $a2
    (fldgt.d, (fa0, a1, a2), "a0987438"),  # fldgt.d $fa0, $a1, $a2
    (fldle.s, (fa0, a1, a2), "a0187538"),  # fldle.s $fa0, $a1, $a2
    (fldle.d, (fa0, a1, a2), "a0987538"),  # fldle.d $fa0, $a1, $a2
    (fstgt.s, (fa0, a1, a2), "a0187638"),  # fstgt.s $fa0, $a1, $a2
    (fstgt.d, (fa0, a1, a2), "a0987638"),  # fstgt.d $fa0, $a1, $a2
    (fstle.s, (fa0, a1, a2), "a0187738"),  # fstle.s $fa0, $a1, $a2
    (fstle.d, (fa0, a1, a2), "a0987738"),  # fstle.d $fa0, $a1, $a2
    (nop, (), "00004003"),  # nop
    (move, (a0, a1), "a4001500"),  # move $a0, $a1
    (ret, (), "2000004c"),  # ret
    (jr, (a1,), "a000004c"),  # jr $a1
    (ud, (5,), "a5046038"),  # ud 5
    (ud, (0,), "00046038"),  # ud 0
    (ud, (31,), "ff076038"),  # ud 31
    (rdcntvl.w, (a0,), "04600000"),  # rdcntvl.w $a0
    (rdcntvh.w, (a0,), "04640000"),  # rdcntvh.w $a0
    (rdcntid.w, (a1,), "a0600000"),  # rdcntid.w $a1
    (li.w, (a0, 0x12345678), "a468241484e09903"),  # li.w $a0, 0x12345678
    (li.w, (t8, -0x80000000), "14000015"),  # li.w $t8, -0x80000000
    (li.w, (s0, 0xffffffff), "17fcbf02"),  # li.w $s0, 0xffffffff
    (li.d, (a0, 0x123456789abcdef0), "a479351584c0bb0304cf8a16848c0403"),  # li.d $a0, 0x123456789abcdef0
    (li.d, (t8, -0x8000000000000000), "14002003"),  # li.d $t8, -0x8000000000000000
    (li.d, (s0, 0xffffffffffffffff), "17fcbf02"),  # li.d $s0, 0xffffffffffffffff
    (fcmp.caf.s, (fcc1, fa1, fa2), "2108100c"),  # fcmp.caf.s $fcc1, $fa1, $fa2
    (fcmp.caf.s, (fcc0, fs7, fa7), "e01f100c"),  # fcmp.caf.s $fcc0, $fs7, $fa7
    (fcmp.caf.s, (fcc7, ft8, fs3), "076e100c"),  # fcmp.caf.s $fcc7, $ft8, $fs3
    (fcmp.saf.s, (fcc1, fa1, fa2), "2188100c"),  # fcmp.saf.s $fcc1, $fa1, $fa2
    (fcmp.saf.s, (fcc0, fs7, fa7), "e09f100c"),  # fcmp.saf.s $fcc0, $fs7, $fa7
    (fcmp.saf.s, (fcc7, ft8, fs3), "07ee100c"),  # fcmp.saf.s $fcc7, $ft8, $fs3
    (fcmp.clt.s, (fcc1, fa1, fa2), "2108110c"),  # fcmp.clt.s $fcc1, $fa1, $fa2
    (fcmp.clt.s, (fcc0, fs7, fa7), "e01f110c"),  # fcmp.clt.s $fcc0, $fs7, $fa7
    (fcmp.clt.s, (fcc7, ft8, fs3), "076e110c"),  # fcmp.clt.s $fcc7, $ft8, $fs3
    (fcmp.slt.s, (fcc1, fa1, fa2), "2188110c"),  # fcmp.slt.s $fcc1, $fa1, $fa2
    (fcmp.slt.s, (fcc0, fs7, fa7), "e09f110c"),  # fcmp.slt.s $fcc0, $fs7, $fa7
    (fcmp.slt.s, (fcc7, ft8, fs3), "07ee110c"),  # fcmp.slt.s $fcc7, $ft8, $fs3
    (fcmp.ceq.s, (fcc1, fa1, fa2), "2108120c"),  # fcmp.ceq.s $fcc1, $fa1, $fa2
    (fcmp.ceq.s, (fcc0, fs7, fa7), "e01f120c"),  # fcmp.ceq.s $fcc0, $fs7, $fa7
    (fcmp.ceq.s, (fcc7, ft8, fs3), "076e120c"),  # fcmp.ceq.s $fcc7, $ft8, $fs3
    (fcmp.seq.s, (fcc1, fa1, fa2), "2188120c"),  # fcmp.seq.s $fcc1, $fa1, $fa2
    (fcmp.seq.s, (fcc0, fs7, fa7), "e09f120c"),  # fcmp.seq.s $fcc0, $fs7, $fa7
    (fcmp.seq.s, (fcc7, ft8, fs3), "07ee120c"),  # fcmp.seq.s $fcc7, $ft8, $fs3
    (fcmp.cle.s, (fcc1, fa1, fa2), "2108130c"),  # fcmp.cle.s $fcc1, $fa1, $fa2
    (fcmp.cle.s, (fcc0, fs7, fa7), "e01f130c"),  # fcmp.cle.s $fcc0, $fs7, $fa7
    (fcmp.cle.s, (fcc7, ft8, fs3), "076e130c"),  # fcmp.cle.s $fcc7, $ft8, $fs3
    (fcmp.sle.s, (fcc1, fa1, fa2), "2188130c"),  # fcmp.sle.s $fcc1, $fa1, $fa2
    (fcmp.sle.s, (fcc0, fs7, fa7), "e09f130c"),  # fcmp.sle.s $fcc0, $fs7, $fa7
    (fcmp.sle.s, (fcc7, ft8, fs3), "07ee130c"),  # fcmp.sle.s $fcc7, $ft8, $fs3
    (fcmp.cun.s, (fcc1, fa1, fa2), "2108140c"),  # fcmp.cun.s $fcc1, $fa1, $fa2
    (fcmp.cun.s, (fcc0, fs7, fa7), "e01f140c"),  # fcmp.cun.s $fcc0, $fs7, $fa7
    (fcmp.cun.s, (fcc7, ft8, fs3), "076e140c"),  # fcmp.cun.s $fcc7, $ft8, $fs3
    (fcmp.sun.s, (fcc1, fa1, fa2), "2188140c"),  # fcmp.sun.s $fcc1, $fa1, $fa2
    (fcmp.sun.s, (fcc0, fs7, fa7), "e09f140c"),  # fcmp.sun.s $fcc0, $fs7, $fa7
    (fcmp.sun.s, (fcc7, ft8, fs3), "07ee140c"),  # fcmp.sun.s $fcc7, $ft8, $fs3
    (fcmp.cult.s, (fcc1, fa1, fa2), "2108150c"),  # fcmp.cult.s $fcc1, $fa1, $fa2
    (fcmp.cult.s, (fcc0, fs7, fa7), "e01f150c"),  # fcmp.cult.s $fcc0, $fs7, $fa7
    (fcmp.cult.s, (fcc7, ft8, fs3), "076e150c"),  # fcmp.cult.s $fcc7, $ft8, $fs3
    (fcmp.sult.s, (fcc1, fa1, fa2), "2188150c"),  # fcmp.sult.s $fcc1, $fa1, $fa2
    (fcmp.sult.s, (fcc0, fs7, fa7), "e09f150c"),  # fcmp.sult.s $fcc0, $fs7, $fa7
    (fcmp.sult.s, (fcc7, ft8, fs3), "07ee150c"),  # fcmp.sult.s $fcc7, $ft8, $fs3
    (fcmp.cueq.s, (fcc1, fa1, fa2), "2108160c"),  # fcmp.cueq.s $fcc1, $fa1, $fa2
    (fcmp.cueq.s, (fcc0, fs7, fa7), "e01f160c"),  # fcmp.cueq.s $fcc0, $fs7, $fa7
    (fcmp.cueq.s, (fcc7, ft8, fs3), "076e160c"),  # fcmp.cueq.s $fcc7, $ft8, $fs3
    (fcmp.sueq.s, (fcc1, fa1, fa2), "2188160c"),  # fcmp.sueq.s $fcc1, $fa1, $fa2
    (fcmp.sueq.s, (fcc0, fs7, fa7), "e09f160c"),  # fcmp.sueq.s $fcc0, $fs7, $fa7
    (fcmp.sueq.s, (fcc7, ft8, fs3), "07ee160c"),  # fcmp.sueq.s $fcc7, $ft8, $fs3
    (fcmp.cule.s, (fcc1, fa1, fa2), "2108170c"),  # fcmp.cule.s $fcc1, $fa1, $fa2
    (fcmp.cule.s, (fcc0, fs7, fa7), "e01f170c"),  # fcmp.cule.s $fcc0, $fs7, $fa7
    (fcmp.cule.s, (fcc7, ft8, fs3), "076e170c"),  # fcmp.cule.s $fcc7, $ft8, $fs3
    (fcmp.sule.s, (fcc1, fa1, fa2), "2188170c"),  # fcmp.sule.s $fcc1, $fa1, $fa2
    (fcmp.sule.s, (fcc0, fs7, fa7), "e09f170c"),  # fcmp.sule.s $fcc0, $fs7, $fa7
    (fcmp.sule.s, (fcc7, ft8, fs3), "07ee170c"),  # fcmp.sule.s $fcc7, $ft8, $fs3
    (fcmp.cne.s, (fcc1, fa1, fa2), "2108180c"),  # fcmp.cne.s $fcc1, $fa1, $fa2
    (fcmp.cne.s, (fcc0, fs7, fa7), "e01f180c"),  # fcmp.cne.s $fcc0, $fs7, $fa7
    (fcmp.cne.s, (fcc7, ft8, fs3), "076e180c"),  # fcmp.cne.s $fcc7, $ft8, $fs3
    (fcmp.sne.s, (fcc1, fa1, fa2), "2188180c"),  # fcmp.sne.s $fcc1, $fa1, $fa2
    (fcmp.sne.s, (fcc0, fs7, fa7), "e09f180c"),  # fcmp.sne.s $fcc0, $fs7, $fa7
    (fcmp.sne.s, (fcc7, ft8, fs3), "07ee180c"),  # fcmp.sne.s $fcc7, $ft8, $fs3
    (fcmp.cor.s, (fcc1, fa1, fa2), "21081a0c"),  # fcmp.cor.s $fcc1, $fa1, $fa2
    (fcmp.cor.s, (fcc0, fs7, fa7), "e01f1a0c"),  # fcmp.cor.s $fcc0, $fs7, $fa7
    (fcmp.cor.s, (fcc7, ft8, fs3), "076e1a0c"),  # fcmp.cor.s $fcc7, $ft8, $fs3
    (fcmp.sor.s, (fcc1, fa1, fa2), "21881a0c"),  # fcmp.sor.s $fcc1, $fa1, $fa2
    (fcmp.sor.s, (fcc0, fs7, fa7), "e09f1a0c"),  # fcmp.sor.s $fcc0, $fs7, $fa7
    (fcmp.sor.s, (fcc7, ft8, fs3), "07ee1a0c"),  # fcmp.sor.s $fcc7, $ft8, $fs3
    (fcmp.cune.s, (fcc1, fa1, fa2), "21081c0c"),  # fcmp.cune.s $fcc1, $fa1, $fa2
    (fcmp.cune.s, (fcc0, fs7, fa7), "e01f1c0c"),  # fcmp.cune.s $fcc0, $fs7, $fa7
    (fcmp.cune.s, (fcc7, ft8, fs3), "076e1c0c"),  # fcmp.cune.s $fcc7, $ft8, $fs3
    (fcmp.sune.s, (fcc1, fa1, fa2), "21881c0c"),  # fcmp.sune.s $fcc1, $fa1, $fa2
    (fcmp.sune.s, (fcc0, fs7, fa7), "e09f1c0c"),  # fcmp.sune.s $fcc0, $fs7, $fa7
    (fcmp.sune.s, (fcc7, ft8, fs3), "07ee1c0c"),  # fcmp.sune.s $fcc7, $ft8, $fs3
    (fcmp.sgt.s, (fcc1, fa2, fa1), "2188110c"),  # fcmp.sgt.s $fcc1, $fa2, $fa1
    (fcmp.sgt.s, (fcc0, fa7, fs7), "e09f110c"),  # fcmp.sgt.s $fcc0, $fa7, $fs7
    (fcmp.sgt.s, (fcc7, fs3, ft8), "07ee110c"),  # fcmp.sgt.s $fcc7, $fs3, $ft8
    (fcmp.sge.s, (fcc1, fa2, fa1), "2188130c"),  # fcmp.sge.s $fcc1, $fa2, $fa1
    (fcmp.sge.s, (fcc0, fa7, fs7), "e09f130c"),  # fcmp.sge.s $fcc0, $fa7, $fs7
    (fcmp.sge.s, (fcc7, fs3, ft8), "07ee130c"),  # fcmp.sge.s $fcc7, $fs3, $ft8
    (fcmp.cugt.s, (fcc1, fa2, fa1), "2108150c"),  # fcmp.cugt.s $fcc1, $fa2, $fa1
    (fcmp.cugt.s, (fcc0, fa7, fs7), "e01f150c"),  # fcmp.cugt.s $fcc0, $fa7, $fs7
    (fcmp.cugt.s, (fcc7, fs3, ft8), "076e150c"),  # fcmp.cugt.s $fcc7, $fs3, $ft8
    (fcmp.cuge.s, (fcc1, fa2, fa1), "2108170c"),  # fcmp.cuge.s $fcc1, $fa2, $fa1
    (fcmp.cuge.s, (fcc0, fa7, fs7), "e01f170c"),  # fcmp.cuge.s $fcc0, $fa7, $fs7
    (fcmp.cuge.s, (fcc7, fs3, ft8), "076e170c"),  # fcmp.cuge.s $fcc7, $fs3, $ft8
    (fcmp.caf.d, (fcc1, fa1, fa2), "2108200c"),  # fcmp.caf.d $fcc1, $fa1, $fa2
    (fcmp.caf.d, (fcc0, fs7, fa7), "e01f200c"),  # fcmp.caf.d $fcc0, $fs7, $fa7
    (fcmp.caf.d, (fcc7, ft8, fs3), "076e200c"),  # fcmp.caf.d $fcc7, $ft8, $fs3
    (fcmp.saf.d, (fcc1, fa1, fa2), "2188200c"),  # fcmp.saf.d $fcc1, $fa1, $fa2
    (fcmp.saf.d, (fcc0, fs7, fa7), "e09f200c"),  # fcmp.saf.d $fcc0, $fs7, $fa7
    (fcmp.saf.d, (fcc7, ft8, fs3), "07ee200c"),  # fcmp.saf.d $fcc7, $ft8, $fs3
    (fcmp.clt.d, (fcc1, fa1, fa2), "2108210c"),  # fcmp.clt.d $fcc1, $fa1, $fa2
    (fcmp.clt.d, (fcc0, fs7, fa7), "e01f210c"),  # fcmp.clt.d $fcc0, $fs7, $fa7
    (fcmp.clt.d, (fcc7, ft8, fs3), "076e210c"),  # fcmp.clt.d $fcc7, $ft8, $fs3
    (fcmp.slt.d, (fcc1, fa1, fa2), "2188210c"),  # fcmp.slt.d $fcc1, $fa1, $fa2
    (fcmp.slt.d, (fcc0, fs7, fa7), "e09f210c"),  # fcmp.slt.d $fcc0, $fs7, $fa7
    (fcmp.slt.d, (fcc7, ft8, fs3), "07ee210c"),  # fcmp.slt.d $fcc7, $ft8, $fs3
    (fcmp.ceq.d, (fcc1, fa1, fa2), "2108220c"),  # fcmp.ceq.d $fcc1, $fa1, $fa2
    (fcmp.ceq.d, (fcc0, fs7, fa7), "e01f220c"),  # fcmp.ceq.d $fcc0, $fs7, $fa7
    (fcmp.ceq.d, (fcc7, ft8, fs3), "076e220c"),  # fcmp.ceq.d $fcc7, $ft8, $fs3
    (fcmp.seq.d, (fcc1, fa1, fa2), "2188220c"),  # fcmp.seq.d $fcc1, $fa1, $fa2
    (fcmp.seq.d, (fcc0, fs7, fa7), "e09f220c"),  # fcmp.seq.d $fcc0, $fs7, $fa7
    (fcmp.seq.d, (fcc7, ft8, fs3), "07ee220c"),  # fcmp.seq.d $fcc7, $ft8, $fs3
    (fcmp.cle.d, (fcc1, fa1, fa2), "2108230c"),  # fcmp.cle.d $fcc1, $fa1, $fa2
    (fcmp.cle.d, (fcc0, fs7, fa7), "e01f230c"),  # fcmp.cle.d $fcc0, $fs7, $fa7
    (fcmp.cle.d, (fcc7, ft8, fs3), "076e230c"),  # fcmp.cle.d $fcc7, $ft8, $fs3
    (fcmp.sle.d, (fcc1, fa1, fa2), "2188230c"),  # fcmp.sle.d $fcc1, $fa1, $fa2
    (fcmp.sle.d, (fcc0, fs7, fa7), "e09f230c"),  # fcmp.sle.d $fcc0, $fs7, $fa7
    (fcmp.sle.d, (fcc7, ft8, fs3), "07ee230c"),  # fcmp.sle.d $fcc7, $ft8, $fs3
    (fcmp.cun.d, (fcc1, fa1, fa2), "2108240c"),  # fcmp.cun.d $fcc1, $fa1, $fa2
    (fcmp.cun.d, (fcc0, fs7, fa7), "e01f240c"),  # fcmp.cun.d $fcc0, $fs7, $fa7
    (fcmp.cun.d, (fcc7, ft8, fs3), "076e240c"),  # fcmp.cun.d $fcc7, $ft8, $fs3
    (fcmp.sun.d, (fcc1, fa1, fa2), "2188240c"),  # fcmp.sun.d $fcc1, $fa1, $fa2
    (fcmp.sun.d, (fcc0, fs7, fa7), "e09f240c"),  # fcmp.sun.d $fcc0, $fs7, $fa7
    (fcmp.sun.d, (fcc7, ft8, fs3), "07ee240c"),  # fcmp.sun.d $fcc7, $ft8, $fs3
    (fcmp.cult.d, (fcc1, fa1, fa2), "2108250c"),  # fcmp.cult.d $fcc1, $fa1, $fa2
    (fcmp.cult.d, (fcc0, fs7, fa7), "e01f250c"),  # fcmp.cult.d $fcc0, $fs7, $fa7
    (fcmp.cult.d, (fcc7, ft8, fs3), "076e250c"),  # fcmp.cult.d $fcc7, $ft8, $fs3
    (fcmp.sult.d, (fcc1, fa1, fa2), "2188250c"),  # fcmp.sult.d $fcc1, $fa1, $fa2
    (fcmp.sult.d, (fcc0, fs7, fa7), "e09f250c"),  # fcmp.sult.d $fcc0, $fs7, $fa7
    (fcmp.sult.d, (fcc7, ft8, fs3), "07ee250c"),  # fcmp.sult.d $fcc7, $ft8, $fs3
    (fcmp.cueq.d, (fcc1, fa1, fa2), "2108260c"),  # fcmp.cueq.d $fcc1, $fa1, $fa2
    (fcmp.cueq.d, (fcc0, fs7, fa7), "e01f260c"),  # fcmp.cueq.d $fcc0, $fs7, $fa7
    (fcmp.cueq.d, (fcc7, ft8, fs3), "076e260c"),  # fcmp.cueq.d $fcc7, $ft8, $fs3
    (fcmp.sueq.d, (fcc1, fa1, fa2), "2188260c"),  # fcmp.sueq.d $fcc1, $fa1, $fa2
    (fcmp.sueq.d, (fcc0, fs7, fa7), "e09f260c"),  # fcmp.sueq.d $fcc0, $fs7, $fa7
    (fcmp.sueq.d, (fcc7, ft8, fs3), "07ee260c"),  # fcmp.sueq.d $fcc7, $ft8, $fs3
    (fcmp.cule.d, (fcc1, fa1, fa2), "2108270c"),  # fcmp.cule.d $fcc1, $fa1, $fa2
    (fcmp.cule.d, (fcc0, fs7, fa7), "e01f270c"),  # fcmp.cule.d $fcc0, $fs7, $fa7
    (fcmp.cule.d, (fcc7, ft8, fs3), "076e270c"),  # fcmp.cule.d $fcc7, $ft8, $fs3
    (fcmp.sule.d, (fcc1, fa1, fa2), "2188270c"),  # fcmp.sule.d $fcc1, $fa1, $fa2
    (fcmp.sule.d, (fcc0, fs7, fa7), "e09f270c"),  # fcmp.sule.d $fcc0, $fs7, $fa7
    (fcmp.sule.d, (fcc7, ft8, fs3), "07ee270c"),  # fcmp.sule.d $fcc7, $ft8, $fs3
    (fcmp.cne.d, (fcc1, fa1, fa2), "2108280c"),  # fcmp.cne.d $fcc1, $fa1, $fa2
    (fcmp.cne.d, (fcc0, fs7, fa7), "e01f280c"),  # fcmp.cne.d $fcc0, $fs7, $fa7
    (fcmp.cne.d, (fcc7, ft8, fs3), "076e280c"),  # fcmp.cne.d $fcc7, $ft8, $fs3
    (fcmp.sne.d, (fcc1, fa1, fa2), "2188280c"),  # fcmp.sne.d $fcc1, $fa1, $fa2
    (fcmp.sne.d, (fcc0, fs7, fa7), "e09f280c"),  # fcmp.sne.d $fcc0, $fs7, $fa7
    (fcmp.sne.d, (fcc7, ft8, fs3), "07ee280c"),  # fcmp.sne.d $fcc7, $ft8, $fs3
    (fcmp.cor.d, (fcc1, fa1, fa2), "21082a0c"),  # fcmp.cor.d $fcc1, $fa1, $fa2
    (fcmp.cor.d, (fcc0, fs7, fa7), "e01f2a0c"),  # fcmp.cor.d $fcc0, $fs7, $fa7
    (fcmp.cor.d, (fcc7, ft8, fs3), "076e2a0c"),  # fcmp.cor.d $fcc7, $ft8, $fs3
    (fcmp.sor.d, (fcc1, fa1, fa2), "21882a0c"),  # fcmp.sor.d $fcc1, $fa1, $fa2
    (fcmp.sor.d, (fcc0, fs7, fa7), "e09f2a0c"),  # fcmp.sor.d $fcc0, $fs7, $fa7
    (fcmp.sor.d, (fcc7, ft8, fs3), "07ee2a0c"),  # fcmp.sor.d $fcc7, $ft8, $fs3
    (fcmp.cune.d, (fcc1, fa1, fa2), "21082c0c"),  # fcmp.cune.d $fcc1, $fa1, $fa2
    (fcmp.cune.d, (fcc0, fs7, fa7), "e01f2c0c"),  # fcmp.cune.d $fcc0, $fs7, $fa7
    (fcmp.cune.d, (fcc7, ft8, fs3), "076e2c0c"),  # fcmp.cune.d $fcc7, $ft8, $fs3
    (fcmp.sune.d, (fcc1, fa1, fa2), "21882c0c"),  # fcmp.sune.d $fcc1, $fa1, $fa2
    (fcmp.sune.d, (fcc0, fs7, fa7), "e09f2c0c"),  # fcmp.sune.d $fcc0, $fs7, $fa7
    (fcmp.sune.d, (fcc7, ft8, fs3), "07ee2c0c"),  # fcmp.sune.d $fcc7, $ft8, $fs3
    (fcmp.sgt.d, (fcc1, fa2, fa1), "2188210c"),  # fcmp.sgt.d $fcc1, $fa2, $fa1
    (fcmp.sgt.d, (fcc0, fa7, fs7), "e09f210c"),  # fcmp.sgt.d $fcc0, $fa7, $fs7
    (fcmp.sgt.d, (fcc7, fs3, ft8), "07ee210c"),  # fcmp.sgt.d $fcc7, $fs3, $ft8
    (fcmp.sge.d, (fcc1, fa2, fa1), "2188230c"),  # fcmp.sge.d $fcc1, $fa2, $fa1
    (fcmp.sge.d, (fcc0, fa7, fs7), "e09f230c"),  # fcmp.sge.d $fcc0, $fa7, $fs7
    (fcmp.sge.d, (fcc7, fs3, ft8), "07ee230c"),  # fcmp.sge.d $fcc7, $fs3, $ft8
    (fcmp.cugt.d, (fcc1, fa2, fa1), "2108250c"),  # fcmp.cugt.d $fcc1, $fa2, $fa1
    (fcmp.cugt.d, (fcc0, fa7, fs7), "e01f250c"),  # fcmp.cugt.d $fcc0, $fa7, $fs7
    (fcmp.cugt.d, (fcc7, fs3, ft8), "076e250c"),  # fcmp.cugt.d $fcc7, $fs3, $ft8
    (fcmp.cuge.d, (fcc1, fa2, fa1), "2108270c"),  # fcmp.cuge.d $fcc1, $fa2, $fa1
    (fcmp.cuge.d, (fcc0, fa7, fs7), "e01f270c"),  # fcmp.cuge.d $fcc0, $fa7, $fs7
    (fcmp.cuge.d, (fcc7, fs3, ft8), "076e270c"),  # fcmp.cuge.d $fcc7, $fs3, $ft8
    (amswap.w, (a0, a2, a1), "a4186038"),  # amswap.w $a0, $a2, $a1
    (amswap.d, (a0, a2, a1), "a4986038"),  # amswap.d $a0, $a2, $a1
    (amswap_db.w, (a0, a2, a1), "a4186938"),  # amswap_db.w $a0, $a2, $a1
    (amswap_db.d, (a0, a2, a1), "a4986938"),  # amswap_db.d $a0, $a2, $a1
    (amadd.w, (a0, a2, a1), "a4186138"),  # amadd.w $a0, $a2, $a1
    (amadd.d, (a0, a2, a1), "a4986138"),  # amadd.d $a0, $a2, $a1
    (amadd_db.w, (a0, a2, a1), "a4186a38"),  # amadd_db.w $a0, $a2, $a1
    (amadd_db.d, (a0, a2, a1), "a4986a38"),  # amadd_db.d $a0, $a2, $a1
    (amand.w, (a0, a2, a1), "a4186238"),  # amand.w $a0, $a2, $a1
    (amand.d, (a0, a2, a1), "a4986238"),  # amand.d $a0, $a2, $a1
    (amand_db.w, (a0, a2, a1), "a4186b38"),  # amand_db.w $a0, $a2, $a1
    (amand_db.d, (a0, a2, a1), "a4986b38"),  # amand_db.d $a0, $a2, $a1
    (amor.w, (a0, a2, a1), "a4186338"),  # amor.w $a0, $a2, $a1
    (amor.d, (a0, a2, a1), "a4986338"),  # amor.d $a0, $a2, $a1
    (amor_db.w, (a0, a2, a1), "a4186c38"),  # amor_db.w $a0, $a2, $a1
    (amor_db.d, (a0, a2, a1), "a4986c38"),  # amor_db.d $a0, $a2, $a1
    (amxor.w, (a0, a2, a1), "a4186438"),  # amxor.w $a0, $a2, $a1
    (amxor.d, (a0, a2, a1), "a4986438"),  # amxor.d $a0, $a2, $a1
    (amxor_db.w, (a0, a2, a1), "a4186d38"),  # amxor_db.w $a0, $a2, $a1
    (amxor_db.d, (a0, a2, a1), "a4986d38"),  # amxor_db.d $a0, $a2, $a1
    (ammax.w, (a0, a2, a1), "a4186538"),  # ammax.w $a0, $a2, $a1
    (ammax.d, (a0, a2, a1), "a4986538"),  # ammax.d $a0, $a2, $a1
    (ammax_db.w, (a0, a2, a1), "a4186e38"),  # ammax_db.w $a0, $a2, $a1
    (ammax_db.d, (a0, a2, a1), "a4986e38"),  # ammax_db.d $a0, $a2, $a1
    (ammin.w, (a0, a2, a1), "a4186638"),  # ammin.w $a0, $a2, $a1
    (ammin.d, (a0, a2, a1), "a4986638"),  # ammin.d $a0, $a2, $a1
    (ammin_db.w, (a0, a2, a1), "a4186f38"),  # ammin_db.w $a0, $a2, $a1
    (ammin_db.d, (a0, a2, a1), "a4986f38"),  # ammin_db.d $a0, $a2, $a1
    (ammax.wu, (a0, a2, a1), "a4186738"),  # ammax.wu $a0, $a2, $a1
    (ammax.du, (a0, a2, a1), "a4986738"),  # ammax.du $a0, $a2, $a1
    (ammax_db.wu, (a0, a2, a1), "a4187038"),  # ammax_db.wu $a0, $a2, $a1
    (ammax_db.du, (a0, a2, a1), "a4987038"),  # ammax_db.du $a0, $a2, $a1
    (ammin.wu, (a0, a2, a1), "a4186838"),  # ammin.wu $a0, $a2, $a1
    (ammin.du, (a0, a2, a1), "a4986838"),  # ammin.du $a0, $a2, $a1
    (ammin_db.wu, (a0, a2, a1), "a4187138"),  # ammin_db.wu $a0, $a2, $a1
    (ammin_db.du, (a0, a2, a1), "a4987138"),  # ammin_db.du $a0, $a2, $a1
]

# Label targets: (fn, ops, backward, forward). Backward: the label is bound
# one nop before the instruction (-4). Forward: the label follows the
# instruction and two nops. Loads and stores to a label are compared with
# the explicit `pcalau12i %pc_hi20; ... %pc_lo12` pair (GNU as has no
# macro for them).
LABEL_CASES = [
    (pcaddi, (a0, LBL), "e4ffff19", "64000018"),  # pcaddi $a0, >1
    (ld.b, (a0, LBL), "0400001a84000028", "0400001a84400028"),  # ld.b $a0, >1
    (ld.h, (a0, LBL), "0400001a84004028", "0400001a84404028"),  # ld.h $a0, >1
    (ld.w, (a0, LBL), "0400001a84008028", "0400001a84408028"),  # ld.w $a0, >1
    (ld.d, (a0, LBL), "0400001a8400c028", "0400001a8440c028"),  # ld.d $a0, >1
    (ld.bu, (a0, LBL), "0400001a8400002a", "0400001a8440002a"),  # ld.bu $a0, >1
    (ld.hu, (a0, LBL), "0400001a8400402a", "0400001a8440402a"),  # ld.hu $a0, >1
    (ld.wu, (a0, LBL), "0400001a8400802a", "0400001a8440802a"),  # ld.wu $a0, >1
    (st.b, (a0, LBL, t0), "0c00001a84010029", "0c00001a84410029"),  # st.b $a0, >1, $t0
    (st.h, (a0, LBL, t0), "0c00001a84014029", "0c00001a84414029"),  # st.h $a0, >1, $t0
    (st.w, (a0, LBL, t0), "0c00001a84018029", "0c00001a84418029"),  # st.w $a0, >1, $t0
    (st.d, (a0, LBL, t0), "0c00001a8401c029", "0c00001a8441c029"),  # st.d $a0, >1, $t0
    (beqz, (t0, LBL), "9ffdff43", "800d0040"),  # beqz $t0, >1
    (bnez, (t0, LBL), "9ffdff47", "800d0044"),  # bnez $t0, >1
    (bceqz, (fcc2, LBL), "5ffcff4b", "400c0048"),  # bceqz $fcc2, >1
    (bcnez, (fcc2, LBL), "5ffdff4b", "400d0048"),  # bcnez $fcc2, >1
    (b, (LBL,), "ffffff53", "000c0050"),  # b >1
    (bl, (LBL,), "ffffff57", "000c0054"),  # bl >1
    (beq, (t0, a0, LBL), "84fdff5b", "840d0058"),  # beq $t0, $a0, >1
    (bne, (t0, a0, LBL), "84fdff5f", "840d005c"),  # bne $t0, $a0, >1
    (blt, (t0, a0, LBL), "84fdff63", "840d0060"),  # blt $t0, $a0, >1
    (bge, (t0, a0, LBL), "84fdff67", "840d0064"),  # bge $t0, $a0, >1
    (bltu, (t0, a0, LBL), "84fdff6b", "840d0068"),  # bltu $t0, $a0, >1
    (bgeu, (t0, a0, LBL), "84fdff6f", "840d006c"),  # bgeu $t0, $a0, >1
    (fld.s, (fa0, LBL, t0), "0c00001a8001002b", "0c00001a8041002b"),  # fld.s $fa0, >1, $t0
    (fst.s, (fa0, LBL, t0), "0c00001a8001402b", "0c00001a8041402b"),  # fst.s $fa0, >1, $t0
    (fld.d, (fa0, LBL, t0), "0c00001a8001802b", "0c00001a8041802b"),  # fld.d $fa0, >1, $t0
    (fst.d, (fa0, LBL, t0), "0c00001a8001c02b", "0c00001a8041c02b"),  # fst.d $fa0, >1, $t0
    (bgt, (a0, t0, LBL), "84fdff63", "840d0060"),  # bgt $a0, $t0, >1
    (ble, (a0, t0, LBL), "84fdff67", "840d0064"),  # ble $a0, $t0, >1
    (bgtu, (a0, t0, LBL), "84fdff6b", "840d0068"),  # bgtu $a0, $t0, >1
    (bleu, (a0, t0, LBL), "84fdff6f", "840d006c"),  # bleu $a0, $t0, >1
    (bltz, (t0, LBL), "80fdff63", "800d0060"),  # bltz $t0, >1
    (bgez, (t0, LBL), "80fdff67", "800d0064"),  # bgez $t0, >1
    (bgtz, (a0, LBL), "04fcff63", "040c0060"),  # bgtz $a0, >1
    (blez, (a0, LBL), "04fcff67", "040c0064"),  # blez $a0, >1
    (la.local, (a0, LBL), "0400001a8400c002", "0400001a8440c002"),  # la.local $a0, >1
    (la.pcrel, (a0, LBL), "0400001a8400c002", "0400001a8440c002"),  # la.pcrel $a0, >1
    (call36, (LBL,), "0100001e21fcff4f", "0100001e2110004c"),  # call36 >1
    (tail36, (t1, LBL), "0d00001ea0fdff4f", "0d00001ea011004c"),  # tail36 $t1, >1
    (call, (LBL,), "0100001e21fcff4f", "0100001e2110004c"),  # call >1
    (tail, (t1, LBL), "0d00001ea0fdff4f", "0d00001ea011004c"),  # tail $t1, >1
]
# fmt: on


_TEXT_NAMES = {v: k for k, v in PY_NAMES.items()}


def mnemonic(fn) -> str:
    name = str(fn.__qualname__)
    return _TEXT_NAMES.get(name, name)


def gnu_text(fn, ops):
    name = mnemonic(fn)
    return name + (" " + ", ".join(str(o) for o in ops) if ops else "")


def _case_id(case):
    return gnu_text(case[0], case[1])


def _subst(ops, lbl):
    return [lbl if isinstance(o, str) and o == LBL else o for o in ops]


@pytest.mark.parametrize("fn, ops, want", CASES, ids=[_case_id(c) for c in CASES])
def test_encode(fn, ops, want):
    assert hexof(fn, *ops) == want


@pytest.mark.parametrize("fn, ops, back, fwd", LABEL_CASES, ids=[_case_id(c) for c in LABEL_CASES])
def test_label_backward(fn, ops, back, fwd):
    a = Assembler(L)
    lbl = a.label()
    a.nop()
    fn(*_subst(ops, lbl), asm=a)
    assert a.link().data[4:].hex() == back


@pytest.mark.parametrize("fn, ops, back, fwd", LABEL_CASES, ids=[_case_id(c) for c in LABEL_CASES])
def test_label_forward(fn, ops, back, fwd):
    a = Assembler(L)
    lbl = Label()
    fn(*_subst(ops, lbl), asm=a)
    a.nop()
    a.nop()
    a.label(lbl)
    assert a.link().data[:-8].hex() == fwd


def test_every_mnemonic_is_tested():
    tested = {mnemonic(c[0]) for c in CASES + LABEL_CASES}
    assert set(MNEMONIC_ARGC) - tested == set()
    # Every operand count of every mnemonic, and every template alternative.
    counts = {(mnemonic(c[0]), len(c[1])) for c in CASES + LABEL_CASES}
    assert {(m, n) for m, ns in MNEMONIC_ARGC.items() for n in ns} - counts == set()
    alts = {k for k, t in MAP_OP.items() if "|" in t}
    labelled = {f"{mnemonic(c[0])}_{len(c[1])}" for c in LABEL_CASES}
    assert alts <= labelled


# -- li.w and li.d ---------------------------------------------------------------


def _sext(v, bits):
    v &= (1 << bits) - 1
    return v - (1 << bits) if v >> (bits - 1) else v


def simulate_li(words, rd):
    """Run an li expansion: lu12i.w, ori, addi.w, or, lu32i.d and lu52i.d
    on one register."""
    regs = [0] * 32
    mask = (1 << 64) - 1
    for w in words:
        d, j, k = w & 31, (w >> 5) & 31, (w >> 10) & 31
        assert d == rd, f"{w:08x} writes r{d}"
        if w >> 25 == 0x14 >> 1:  # lu12i.w
            regs[d] = _sext((w >> 5 & 0xFFFFF) << 12, 32)
        elif w >> 22 == 0x03800000 >> 22:  # ori
            regs[d] = regs[j] | (w >> 10 & 0xFFF)
        elif w >> 22 == 0x02800000 >> 22:  # addi.w
            regs[d] = _sext(regs[j] + _sext(w >> 10, 12), 32)
        elif w >> 15 == 0x00150000 >> 15:  # or
            regs[d] = regs[j] | regs[k]
        elif w >> 25 == 0x16 >> 1:  # lu32i.d
            regs[d] = _sext((regs[d] & 0xFFFFFFFF) | (w >> 5 & 0xFFFFF) << 32, 52)
        elif w >> 22 == 0x03000000 >> 22:  # lu52i.d
            regs[d] = _sext((regs[j] & ((1 << 52) - 1)) | (w >> 10 & 0xFFF) << 52, 64)
        else:
            raise AssertionError(f"unexpected word {w:08x} in li")
        regs[0] = 0
    return regs[rd] & mask


def _li_values(bits):
    rng = random.Random(1234 + bits)
    top = 1 << bits
    vals = {0, 1, -1, 0x7FF, 0x800, 0xFFF, 0x1000, -0x800, -0x801, 0x7FFFF000, 0x80000000, 0xFFFFFFFF,
            -0x80000000, 0x7FFFFFFF, 0xFFFFF000, 0xFFFFF7FF, 0xFFFFF800}  # fmt: skip
    for bit in range(bits):
        for v in (1 << bit, (1 << bit) - 1, -(1 << bit), (1 << bit) + 0x800, (1 << bit) - 0x801, ~(1 << bit)):
            vals.add(v)
    # Values built from the four parts, each zero, all ones, a sign bit
    # alone or random, hit every combination of parts left out.
    choices = [(0, 0xFFF, 0x800, 0x7FF), (0, 0xFFFFF, 0x80000, 0x7FFFF), (0, 0xFFFFF, 0x80000, 0x7FFFF), (0, 0xFFF, 0x800, 0x7FF)]
    for picks in itertools.product(range(5), repeat=4):
        v = 0
        for i, (shift, width, c) in enumerate(zip((0, 12, 32, 52), (12, 20, 20, 12), picks)):
            part = rng.getrandbits(width) if c == 4 else choices[i][c]
            v |= part << shift
        vals.add(v & (top - 1))
    for _ in range(2000):
        vals.add(rng.getrandbits(bits))
        vals.add(rng.getrandbits(rng.randrange(1, bits + 1)) - (rng.getrandbits(1) << (bits - 1)))
        vals.add(sum(1 << rng.randrange(bits) for _ in range(rng.randrange(1, 4))))
    return sorted(v for v in vals if -(1 << (bits - 1)) <= v < top)


LI_D_VALUES = _li_values(64)
LI_W_VALUES = _li_values(32)


def test_li_simulated():
    for v in LI_D_VALUES:
        for rd in (4, 31):
            words = li_words(rd, v)
            assert 1 <= len(words) <= 4, (hex(v), len(words))
            assert simulate_li(words, rd) == v & ((1 << 64) - 1), hex(v)
    for v in LI_W_VALUES:
        words = li_words(4, v, 32)
        assert 1 <= len(words) <= 2, (hex(v), len(words))
        assert simulate_li(words, 4) == _sext(v, 32) & ((1 << 64) - 1), hex(v)


def test_li_lengths():
    assert li_words(4, 0) == [0x00150004]  # or $a0, $zero, $zero
    assert len(li_words(4, 5)) == 1 and len(li_words(4, -5)) == 1
    assert len(li_words(4, 0x12345000)) == 1  # lu12i.w alone
    assert len(li_words(4, 0x12345678)) == 2
    assert len(li_words(4, 0x0010000000000000)) == 1  # lu52i.d from zero
    assert len(li_words(4, 0x123456789ABCDEF0)) == 4
    assert max(len(li_words(4, v)) for v in LI_D_VALUES) == 4


def test_li_range_and_listing():
    a = Assembler(L)
    a.li.d(a0, 0x123456789ABCDEF0)  # noqa: F405
    a.li.w(a1, 0xFFFFFFFF)  # noqa: F405
    assert a.cur.insns == [(0, 16, "li.d", (a0, 0x123456789ABCDEF0)), (16, 20, "li.w", (a1, 0xFFFFFFFF))]  # noqa: F405
    for bad in (1 << 64, -(1 << 63) - 1):
        with pytest.raises(EncodeError, match="64 bit"):
            a.li.d(a0, bad)  # noqa: F405
    # GNU as cuts li.w values whose high 32 bits are all ones to 32 bits
    # (li.w $a0, -0x80000001 is 0x7fffffff); jita rejects them.
    for bad in (1 << 32, -(1 << 31) - 1):
        with pytest.raises(EncodeError, match="32 bit"):
            a.li.w(a0, bad)  # noqa: F405
    with pytest.raises(EncodeError, match="expected an immediate"):
        a.li.d(a0, "x")  # noqa: F405
    with pytest.raises(EncodeError, match="li is not an instruction"):
        a.li(a0, 1)  # noqa: F405


@requires_loongarch64_oracle
@pytest.mark.parametrize("width", ["d", "w"])
def test_li_matches_gnu(width):
    # One assembly for all values, compared word by word per value.
    values = LI_D_VALUES if width == "d" else LI_W_VALUES
    code = assemble("\n".join(f"li.{width} $a0, {v}" for v in values), arch="loongarch64")
    pos = 0
    for v in values:
        want = b"".join(w.to_bytes(4, "little") for w in li_words(4, v, 64 if width == "d" else 32))
        assert code[pos : pos + len(want)] == want, hex(v)
        pos += len(want)
    assert pos == len(code)


# -- labels, relocations, externs ------------------------------------------------


def test_string_labels_and_methods():
    a = Assembler(L)
    a.label("top")
    a.addi.d(a0, a0, -1)  # noqa: F405
    a.bnez(a0, "top")  # noqa: F405
    a.beqz(a0, "out")  # noqa: F405
    a.b("top")
    a.label("out")
    a.ret()
    assert a.link().data.hex() == "84fcff02" + "9ffcff47" + "80080040" + "fff7ff53" + "2000004c"


@pytest.mark.parametrize(
    "fn, ops, limit",
    [
        (beq, (a0, a1), 1 << 17),  # noqa: F405
        (bgtz, (a0,), 1 << 17),  # noqa: F405
        (beqz, (a0,), 1 << 22),  # noqa: F405
        (bcnez, (fcc0,), 1 << 22),  # noqa: F405
        (b, (), 1 << 27),  # noqa: F405
        (bl, (), 1 << 27),  # noqa: F405
        (pcaddi, (a0,), 1 << 21),  # noqa: F405
    ],
)
def test_branch_ranges(fn, ops, limit):
    base = 1 << 32
    a = Assembler(L)
    fn(*ops, Extern("t"), asm=a)
    for n in (limit - 4, -limit, 4, -4, 0x7FFC, -0x8000, 0x10000, -0x10004):
        if -limit <= n < limit:
            word = int.from_bytes(a.link(base=base, externs={"t": base + n}).data, "little")
            assert _branch_offset(word) == n
    with pytest.raises(LinkError, match="out of range"):
        a.link(base=base, externs={"t": base + limit})
    with pytest.raises(LinkError, match="out of range"):
        a.link(base=base, externs={"t": base - limit - 4})
    with pytest.raises(LinkError, match="multiple of 4"):
        a.link(base=base, externs={"t": base + 2})


def _branch_offset(word):
    """The byte offset of a branch or pcaddi word, decoded by opcode."""
    op = word >> 26
    if op in (0x10, 0x11, 0x12):  # beqz, bnez, bceqz/bcnez
        return _sext((word >> 10 & 0xFFFF) | (word & 31) << 16, 21) * 4
    if op in (0x14, 0x15):  # b, bl
        return _sext((word >> 10 & 0xFFFF) | (word & 0x3FF) << 16, 26) * 4
    if word >> 25 == 0x0C:  # pcaddi
        return _sext(word >> 5, 20) * 4
    return _sext(word >> 10, 16) * 4


def _pcala_address(code, place):
    """What pcalau12i + (addi.d or a load) at `place` compute."""
    first, second = int.from_bytes(code[:4], "little"), int.from_bytes(code[4:8], "little")
    return (place & ~0xFFF) + (_sext(first >> 5, 20) << 12) + _sext(second >> 10, 12)


def _call36_address(code, place):
    first, second = int.from_bytes(code[:4], "little"), int.from_bytes(code[4:8], "little")
    return place + (_sext(first >> 5, 20) << 18) + (_sext(second >> 10, 16) << 2)


def _pcala_limits(place):
    page = place >> 12
    return ((page - 0x80000) << 12) - 0x800, ((page + 0x80000) << 12) - 0x801


@pytest.mark.parametrize("offset", [0, 4, 0x7FC, 0x800, 0xFFC])
@pytest.mark.parametrize("fn, ops", [(la.local, (a0,)), (ld.d, (a0,)), (st.w, (a0, t0))])  # noqa: F405
def test_pcala_ranges(fn, ops, offset):
    # place anywhere in its page; targets around the 0x800 rounding, page
    # boundaries and both ends of the +-2GB range.
    base = 1 << 36
    place = base + offset
    a = Assembler(L)
    for _ in range(offset // 4):
        a.nop()
    fn(*ops[:1], Extern("t"), *ops[1:], asm=a)
    lo, hi = _pcala_limits(place)
    targets = [lo, hi, place, place + 0x7FF, place + 0x800, place - 0x800, place - 0x801, place + 0x1234_5678]
    targets += [base + d for d in (0, 0x7FF, 0x800, 0xFFF, 0x1000, -1, -0x800, -0x801)]
    for t in targets:
        code = a.link(base=base, externs={"t": t}).data[offset:]
        assert _pcala_address(code, place) == t, hex(t)
    for t in (lo - 1, hi + 1):
        with pytest.raises(LinkError, match="out of range"):
            a.link(base=base, externs={"t": t})


CALL36_MAX = ((1 << 19) - 1 << 18) + 0x1FFFC
CALL36_MIN = -(1 << 37) - 0x20000


@pytest.mark.parametrize("fn, ops", [(call36, ()), (tail36, (t0,)), (call, ()), (tail, (t1,))])  # noqa: F405
def test_call36_ranges(fn, ops):
    base = 1 << 40
    a = Assembler(L)
    fn(*ops, Extern("t"), asm=a)
    for n in (CALL36_MAX, CALL36_MIN, 0, 4, -4, 0x1FFFC, 0x20000, -0x20000, -0x20004, 0x12_3456_789C):
        code = a.link(base=base, externs={"t": base + n}).data
        assert _call36_address(code, base) == base + n, hex(n)
    for n in (CALL36_MAX + 4, CALL36_MIN - 4):
        with pytest.raises(LinkError, match="out of range"):
            a.link(base=base, externs={"t": base + n})
    with pytest.raises(LinkError, match="multiple of 4"):
        a.link(base=base, externs={"t": base + 2})


def test_split_helpers():
    assert split_pcala(0x1800, 0x1000) == (1, -0x800)
    assert split_pcala(0x17FF, 0x1FFC) == (0, 0x7FF)
    assert split_pcala(0x0, 0xFFC) == (0, 0)
    assert split_pcala(0x800, 0) == (1, -0x800)
    assert split_pcala(_pcala_limits(0x5000)[1], 0x5000) == (0x7FFFF, 0x7FF)
    assert split_pcala(_pcala_limits(0x5000)[0], 0x5000) == (-0x80000, -0x800)
    assert split_pcala(_pcala_limits(0x5000)[1] + 1, 0x5000) is None
    assert split_call36(0x20000) == (1, -0x8000)
    assert split_call36(0x1FFFC) == (0, 0x7FFF)
    assert split_call36(CALL36_MAX) == (0x7FFFF, 0x7FFF)
    assert split_call36(CALL36_MIN) == (-0x80000, -0x8000)
    assert split_call36(CALL36_MAX + 4) is None


def test_patch_kinds_or_into_words():
    buf = bytearray(bytes.fromhex("00000058"))
    B16.apply(buf, 0, 0x1000 - 4, 0x1000)
    assert buf.hex() == "00fcff5b"
    buf = bytearray(bytes.fromhex("00000040"))
    B21.apply(buf, 0, 0x1000 + 0x3FFFFC, 0x1000)
    assert buf.hex() == "0ffcff43"
    buf = bytearray(bytes.fromhex("00000050"))
    B26.apply(buf, 0, 0x1000 - (1 << 27), 0x1000)
    assert buf.hex() == "00020050"
    buf = bytearray(bytes.fromhex("04000018"))
    PCREL20.apply(buf, 0, 0x1000 - 4, 0x1000)
    assert buf.hex() == "e4ffff19"
    buf = bytearray(bytes.fromhex("0400001a" "8400c002"))
    PCALA.apply(buf, 0, 0x2000 + 0x12345678, 0x2000)
    assert _pcala_address(buf, 0x2000) == 0x2000 + 0x12345678
    buf = bytearray(bytes.fromhex("0100001e" "2100004c"))
    CALL36.apply(buf, 0, 0x2000 + 0x12345678, 0x2000)
    assert _call36_address(buf, 0x2000) == 0x2000 + 0x12345678
    assert (B16.size, B21.size, B26.size, PCREL20.size, PCALA.size, CALL36.size) == (4, 4, 4, 4, 8, 8)


def test_extern_targets_and_slots():
    base = 0x10000
    a = Assembler(L)
    call36(Extern("f"), asm=a)  # noqa: F405
    ld.d(a0, Extern("g"), asm=a)  # noqa: F405 - loads the 8 bytes at g
    ld.d(t0, a.extern_slot(Extern("h")), asm=a)  # noqa: F405 - loads h's address
    jirl(ra, t0, 0, asm=a)  # noqa: F405
    img = a.link(base=base, externs={"f": base + 0x800, "g": base - 8, "h": 0x7F00_0000_1234})
    assert _call36_address(img.data[0:8], base) == base + 0x800
    assert _pcala_address(img.data[8:16], base + 8) == base - 8
    slot = img.section_offsets["externs"]
    assert _pcala_address(img.data[16:24], base + 16) == base + slot
    assert img.data[slot : slot + 8] == (0x7F00_0000_1234).to_bytes(8, "little")
    with pytest.raises(LinkError, match="out of range"):
        a.link(base=base, externs={"f": base + (1 << 40), "g": 0, "h": 0})


def test_data_directives_with_labels():
    a = Assembler(L)
    lbl = a.label("here")
    a.qword(lbl)
    a.dword(0x12345678)
    assert a.link(base=0x10000).data.hex() == "0000010000000000" + "78563412"


# -- arch object ---------------------------------------------------------------


def test_nop_fill_and_align():
    assert L.NOP == bytes.fromhex("00004003")
    assert L.ARCH.nop_fill(8) == L.NOP * 2
    assert L.ARCH.nop_fill(6) == b"\0\0" + L.NOP
    assert len(L.ARCH.nop_fill(0)) == 0
    a = Assembler(L)
    a.ret()
    a.align(16)
    assert bytes(a.cur.buf) == bytes.fromhex("2000004c") + L.NOP * 3


def test_arch_attributes():
    assert L.ARCH.name == "loongarch64" and L.ARCH.pointer_size == 8 and L.ARCH.imm_prefix == ""
    assert L.ARCH.insns is INSNS
    assert Assembler(L).arch is L.ARCH and Assembler(L.ARCH).arch is L.ARCH
    assert Assembler("loongarch64").arch is L.ARCH
    assert L.IBAR_RET == bytes.fromhex("00807238" "2000004c")  # ibar 0; ret


def test_icache_flush(monkeypatch):
    calls = []
    arch = L.Loongarch64Arch()
    monkeypatch.setattr(L, "_find_flush", lambda: lambda start, end: calls.append((start, end)))
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")
    arch.icache_flush(0x1000, 64)
    assert calls == []
    monkeypatch.setattr(platform, "machine", lambda: "loongarch64")
    arch.icache_flush(0x1000, 64)
    arch.icache_flush(0x2000, 0)
    assert calls == [(0x1000, 0x1040)]


def test_icache_flush_falls_back_to_a_stub(monkeypatch):
    import ctypes

    class FakeLib:
        # No library has __clear_cache.
        def __init__(self, name):
            raise OSError(name)

    marker = object()
    monkeypatch.setattr(ctypes, "CDLL", FakeLib)
    monkeypatch.setattr(L, "_stub_flush", lambda: marker)
    assert L._find_flush() is marker


@pytest.mark.parametrize("machine", ["loongarch64", "LOONGARCH64"])
def test_host_arch(monkeypatch, machine):
    import jita.core.assembler as asm_mod

    monkeypatch.setattr(asm_mod.platform, "machine", lambda: machine)
    assert Assembler().arch is L.ARCH


def test_star_exports():
    names = set(L.__all__)
    assert {"and_", "or_", "break_", "add", "ld", "fcmp", "amadd_db", "li", "la", "call36", "revb", "r0",
            "zero", "fp", "s9", "r21", "f31", "fcc7", "fcsr3", "ARCH", "Loongarch64Arch",
            "Loongarch64Assembler", "label"} <= names  # fmt: skip
    assert not {"and", "or", "break", "encoder", "ctypes", "platform", "INSNS", "Group", "mem"} & names


def test_dotted_mnemonics():
    assert isinstance(add, Group) and isinstance(fcmp.clt, Group) and isinstance(crc.w.b, Group)  # noqa: F405
    assert isinstance(revb, Group) and isinstance(la, Group) and isinstance(li, Group)  # noqa: F405
    assert add.d is MNEMONICS["add.d"] and fcmp.clt.d is MNEMONICS["fcmp.clt.d"]  # noqa: F405
    assert crc.w.b.w is MNEMONICS["crc.w.b.w"] and amadd_db.d is MNEMONICS["amadd_db.d"]  # noqa: F405
    assert revb._2h is MNEMONICS["revb.2h"] and bitrev._8b is MNEMONICS["bitrev.8b"]  # noqa: F405
    assert add.d.__name__ == "d" and add.d.__qualname__ == "add.d"  # noqa: F405
    assert revb._2h.__name__ == "_2h" and revb._2h.__qualname__ == "revb.2h"  # noqa: F405
    with pytest.raises(EncodeError, match="add is not an instruction, use add.d, add.w"):
        add(a0, a1, a2, asm=Assembler(L))  # noqa: F405
    with pytest.raises(EncodeError, match=r"revb is not an instruction, use revb._2h \(revb.2h\), revb._2w \(revb.2w\), revb._4h \(revb.4h\), revb.d$"):
        revb(a0, a1, asm=Assembler(L))  # noqa: F405
    a = Assembler(L)
    a.add.d(a0, a1, a2)  # noqa: F405
    a.fcmp.clt.d(fcc0, fa0, fa1)  # noqa: F405
    a.revb._2h(a0, a1)  # noqa: F405
    a.break_(0)
    a.and_(a0, a1, a2)  # noqa: F405
    assert bytes(a.cur.buf).hex() == "a4981000" "0004210c" "a4300000" "00002a00" "a4981400"
    assert all(getattr(f, "__test__", True) is False for f in MNEMONICS.values())


def test_listing():
    from jita.tools.listing import listing

    a = Assembler(L)
    with a:
        label("loop")
        ld.d(a1, a0, 8)  # noqa: F405
        slli.d(a1, a1, 3)  # noqa: F405
        lu12i.w(a2, -1)  # noqa: F405
        andi(a3, a3, 0xFF)  # noqa: F405
        fcmp.clt.d(fcc1, fa0, fa1)  # noqa: F405
        amadd_db.w(a0, a1, a2)  # noqa: F405
        bnez(a0, "loop")  # noqa: F405
        li.d(t0, 0x12345678)  # noqa: F405
        la.local(a0, "loop")  # noqa: F405
        call36("loop")  # noqa: F405
    text = listing(a)
    assert "ld.d $a1, $a0, 8" in text
    assert "slli.d $a1, $a1, 0x3" in text
    assert "lu12i.w $a2, -1" in text
    assert "andi $a3, $a3, 0xff" in text
    assert "fcmp.clt.d $fcc1, $fa0, $fa1" in text
    assert "amadd_db.w $a0, $a1, $a2" in text
    assert "bnez $a0, loop" in text and "b21 -> loop" in text
    assert "li.d $t0, 0x12345678" in text
    assert "la.local $a0, loop" in text and "pcala -> loop" in text
    assert "call36 loop" in text and "call36 -> loop" in text


# The aliases and pseudo instructions objdump -M no-aliases shows as their
# base instruction.
_ALIASES = {
    "nop", "move", "ret", "jr", "ud", "rdcntvl.w", "rdcntvh.w", "rdcntid.w", "li.w", "li.d",
    "fcmp.sgt.s", "fcmp.sge.s", "fcmp.cugt.s", "fcmp.cuge.s", "fcmp.sgt.d", "fcmp.sge.d",
    "fcmp.cugt.d", "fcmp.cuge.d",
}  # fmt: skip


def _emit(a, fn, ops):
    fn(*ops, asm=a)


def _listing_lines(a):
    from jita.tools.listing import listing

    return [" ".join(line.split("  ")[-1].split()) for line in listing(a).splitlines()[1:]]


@requires_loongarch64_oracle
def test_listing_matches_objdump():
    # Every base instruction of CASES prints as objdump -M no-aliases
    # prints it; the single word aliases as objdump prints them by default.
    base = [c for c in CASES if mnemonic(c[0]) not in _ALIASES]
    a = Assembler(L)
    for fn, ops, _ in base:
        _emit(a, fn, ops)
    assert _listing_lines(a) == loongarch64_disassemble(bytes(a.cur.buf))
    aliases = [c for c in CASES if mnemonic(c[0]) in _ALIASES and len(bytes.fromhex(c[2])) == 4]
    aliases = [c for c in aliases if not mnemonic(c[0]).startswith(("fcmp", "li"))]
    a = Assembler(L)
    for fn, ops, _ in aliases:
        _emit(a, fn, ops)
    assert _listing_lines(a) == loongarch64_disassemble(bytes(a.cur.buf), aliases=True)


# -- operands ------------------------------------------------------------------


def test_register_objects():
    assert r4 is a0 and f0 is fa0 and r22 is fp and s9 is fp and r3 is sp and r0 is zero  # noqa: F405
    assert gpr(3) is sp and fpr(31) is fs7 and gpr(0) is zero and gpr(21) is r21  # noqa: F405
    assert (a0.kind, a0.code, fa0.kind, fa0.code) == ("gp", 4, "fp", 0)  # noqa: F405
    assert (fcc7.kind, fcc7.code, fcsr3.kind, fcsr3.code) == ("fcc", 7, "fcsr", 3)  # noqa: F405
    assert str(r4) == "$a0" and repr(r4) == "a0" and str(f24) == "$fs0" and str(r21) == "$r21"  # noqa: F405
    assert isinstance(a0, R) and isinstance(fa0, F) and isinstance(fcc0, Fcc) and isinstance(fcsr0, Fcsr)  # noqa: F405
    with pytest.raises(EncodeError):
        gpr(32)  # noqa: F405
    with pytest.raises(EncodeError):
        fpr(-1)  # noqa: F405


@requires_loongarch64_oracle
@pytest.mark.parametrize("n", range(32))
def test_every_register_number(n):
    r1, r2, r3 = gpr(n), gpr((n + 7) % 32), gpr((n + 19) % 32)  # noqa: F405
    f1, f2, f3, f4 = fpr(n), fpr((n + 5) % 32), fpr((n + 11) % 32), fpr((n + 29) % 32)  # noqa: F405
    cc = fcc0, fcc1, fcc2, fcc3, fcc4, fcc5, fcc6, fcc7  # noqa: F405
    a = Assembler(L)
    with a:
        add.d(r1, r2, r3)  # noqa: F405
        st.d(r1, r2, -8)  # noqa: F405
        fmadd.d(f1, f2, f3, f4)  # noqa: F405
        movgr2fr.d(f1, r2)  # noqa: F405
        fsel(f1, f2, f3, cc[n % 8])  # noqa: F405
    text = (
        f"add.d {r1}, {r2}, {r3}\nst.d {r1}, {r2}, -8\nfmadd.d {f1}, {f2}, {f3}, {f4}\n"
        f"movgr2fr.d {f1}, {r2}\nfsel {f1}, {f2}, {f3}, {cc[n % 8]}"
    )
    assert bytes(a.cur.buf) == assemble(text, arch="loongarch64")


# -- encoder errors ------------------------------------------------------------


@pytest.mark.parametrize(
    "fn, ops, match",
    [
        (addi.d, (a0, a1, 2048), "signed 12 bit"),  # noqa: F405
        (addi.w, (a0, a1, -2049), "signed 12 bit"),  # noqa: F405
        (andi, (a0, a1, -1), "unsigned 12 bit"),  # noqa: F405
        (ori, (a0, a1, 4096), "unsigned 12 bit"),  # noqa: F405
        (addu16i.d, (a0, a1, 0x8000), "signed 16 bit"),  # noqa: F405
        (lu12i.w, (a0, 0x80000), "signed 20 bit"),  # noqa: F405
        (lu32i.d, (a0, -0x80001), "signed 20 bit"),  # noqa: F405
        (pcaddi, (a0, 0xFFFFF), "signed 20 bit"),  # noqa: F405
        (addi.d, (a0, fa1, 1), "integer register"),  # noqa: F405
        (add.d, (a0, a1, 1), "integer register"),  # noqa: F405
        (fadd.d, (fa0, a1, fa2), "floating point register"),  # noqa: F405
        (fcmp.clt.s, (a0, fa0, fa1), "condition flag register"),  # noqa: F405
        (movgr2fcsr, (a0, a0), "fcsr register"),  # noqa: F405
        (slli.w, (a0, a1, 32), "unsigned 5 bit"),  # noqa: F405
        (slli.d, (a0, a1, 64), "unsigned 6 bit"),  # noqa: F405
        (srai.d, (a0, a1, -1), "unsigned 6 bit"),  # noqa: F405
        (bstrins.d, (a0, a1, 3, 5), "msb 3 is below lsb 5"),  # noqa: F405
        (bstrpick.w, (a0, a1, 32, 0), "unsigned 5 bit"),  # noqa: F405
        (alsl.d, (a0, a1, a2, 0), "shift 1..4"),  # noqa: F405
        (alsl.w, (a0, a1, a2, 5), "shift 1..4"),  # noqa: F405
        (bytepick.w, (a0, a1, a2, 4), "unsigned 2 bit"),  # noqa: F405
        (bytepick.d, (a0, a1, a2, 8), "unsigned 3 bit"),  # noqa: F405
        (ll.w, (a0, a1, 6), "multiple of 4"),  # noqa: F405
        (ldptr.d, (a0, a1, 32768), "out of range"),  # noqa: F405
        (stptr.w, (a0, a1, -32772), "out of range"),  # noqa: F405
        (jirl, (ra, a0, 2), "multiple of 4"),  # noqa: F405
        (jirl, (ra, a0, 131072), "out of range"),  # noqa: F405
        (dbar, (0x8000,), "unsigned 15 bit"),  # noqa: F405
        (break_, (-1,), "unsigned 15 bit"),  # noqa: F405
        (preld, (32, a0, 0), "unsigned 5 bit"),  # noqa: F405
        (ud, (32,), "unsigned 5 bit"),  # noqa: F405
        (ld.d, (a0, a1, 2048), "signed 12 bit"),  # noqa: F405
        (ld.d, (a0, a1), "expected a label"),  # noqa: F405
        (ld.d, (a0, 8), "expected a label"),  # noqa: F405
        (st.d, (a0, a1), "expects 3 operands"),  # noqa: F405
        (amadd.d, (a0, a0, a1), "rd must differ"),  # noqa: F405
        (amadd_db.w, (a0, a1, a0), "rd must differ"),  # noqa: F405
        (ammin_db.du, (t0, t0, a1), "rd must differ"),  # noqa: F405
        (beq, (a0, a1, 8), "expected a label"),  # noqa: F405
        (b, (a0,), "expected a label"),  # noqa: F405
        (call36, (a0,), "expected a label"),  # noqa: F405
        (ret, (a0,), "expects 0 operands"),  # noqa: F405
        (add.d, (a0, a1), "expects 3 operands"),  # noqa: F405
        (movfr2gr.d, (fa0, a0), "integer register"),  # noqa: F405
        (ld.d, (zero, "x"), "zero cannot hold the address"),  # noqa: F405
        (ld.w, (zero, Extern("x")), "zero cannot hold the address"),  # noqa: F405
        (st.d, (a0, "x", zero), "zero cannot hold the address"),  # noqa: F405
        (fld.d, (fa0, "x", zero), "zero cannot hold the address"),  # noqa: F405
        (tail36, (zero, "x"), "zero cannot hold the address"),  # noqa: F405
        (tail, (r0, "x"), "zero cannot hold the address"),  # noqa: F405
        (st.d, (t0, "x", t0), "must differ from the register stored"),  # noqa: F405
        (st.b, (a1, Extern("x"), a1), "must differ from the register stored"),  # noqa: F405
    ],
    ids=lambda v: mnemonic(v) if callable(v) else None,
)
def test_encode_errors(fn, ops, match):
    a = Assembler(L)
    with pytest.raises(EncodeError, match=match):
        fn(*ops, asm=a)
    assert a.pos() == 0 and not a.cur.patches


def test_rejected_instruction_leaves_no_named_label():
    a = Assembler(L)
    with pytest.raises(EncodeError):
        a.st.d(fa0, "z", a0)  # noqa: F405
    a.bind(Label("z"))


def test_unknown_instruction():
    from jita.loongarch64.encoder import encode as raw

    with pytest.raises(EncodeError, match="unknown instruction"):
        raw(Assembler(L), "frobnicate", ())
    assert isinstance(la, Group)  # noqa: F405
    with pytest.raises(EncodeError, match="la is not an instruction, use la.local, la.pcrel"):
        la(a0, "x", asm=Assembler(L))  # noqa: F405


def test_store_to_label_takes_another_register():
    # `st.d $t0, lbl, $t0` would store the pcalau12i result and is
    # rejected (test_encode_errors); a float store has no such clash.
    a = Assembler(L)
    fst.d(ft12, "x", t4, asm=a)  # noqa: F405
    a.label("x")
    assert a.link().data.hex() == "1000001a" "1422c02b"


def test_zero_destinations_that_are_accepted():
    # GNU as takes `la.local $zero, lbl` (nothing is left behind) and the
    # am* instructions with rd = zero; amswap.w alone may repeat rd (ud).
    a = Assembler(L)
    la.local(zero, "x", asm=a)  # noqa: F405
    amadd.d(zero, a1, a1, asm=a)  # noqa: F405
    amswap.w(a0, a1, a0, asm=a)  # noqa: F405
    a.label("x")
    assert a.link().data.hex() == "0000001a" "0040c002" "a0946138" "84146038"


@requires_loongarch64_oracle
@pytest.mark.parametrize(
    "text",
    ["amadd.d $a0, $a0, $a1", "amxor_db.w $a0, $a1, $a0", "bstrpick.d $a0, $a1, 3, 5", "move $u0, $a0"],
)
def test_oracle_rejects(text):
    with pytest.raises(OracleError):
        assemble(text, arch="loongarch64")


@requires_loongarch64_oracle
def test_oracle_accepts_what_jita_rejects():
    # GNU as assembles a tail36 through $zero, which jumps to the low bits
    # of the displacement as an absolute address; jita refuses it.
    assert assemble("tail36 $zero, 1f\n1: nop", arch="loongarch64")
    # And it cuts li.w values whose high 32 bits are all ones to 32 bits.
    assert assemble("li.w $a0, -0x80000001", arch="loongarch64") == b"".join(
        w.to_bytes(4, "little") for w in li_words(4, 0x7FFFFFFF, 32)
    )


# -- oracle --------------------------------------------------------------------


@requires_loongarch64_oracle
def test_oracle():
    # All of CASES in one assembly, split by the expected lengths.
    code = assemble("\n".join(gnu_text(fn, ops) for fn, ops, _ in CASES), arch="loongarch64")
    pos, bad = 0, []
    for fn, ops, want in CASES:
        got = code[pos : pos + len(want) // 2].hex()
        pos += len(want) // 2
        if got != want:
            bad.append(f"{gnu_text(fn, ops)}: GNU {got}, table {want}")
    assert pos == len(code) and not bad


def _label_text(fn, ops, target):
    name = mnemonic(fn)
    if name.startswith(("ld.", "st.", "fld.", "fst.")):
        rd, tmp = ops[0], ops[2] if len(ops) == 3 else ops[0]
        return f"pcalau12i {tmp}, %pc_hi20({target})\n{name} {rd}, {tmp}, %pc_lo12({target})"
    return gnu_text(fn, [target if o == LBL else o for o in ops])


@requires_loongarch64_oracle
@pytest.mark.parametrize("fn, ops, back, fwd", LABEL_CASES, ids=[_case_id(c) for c in LABEL_CASES])
def test_oracle_labels(fn, ops, back, fwd):
    # A global label leaves the reference to the linker (see oracle.py).
    code = assemble(".globl T\nT:\nnop\n" + _label_text(fn, ops, "T"), arch="loongarch64")
    assert code[4:].hex() == back
    code = assemble(_label_text(fn, ops, "T") + "\nnop\nnop\n.globl T\nT:", arch="loongarch64")
    assert code[:-8].hex() == fwd


# Immediate template letters: (bits, signed).
_IMM_FIELDS = {
    "i": (12, True), "u": (12, False), "h": (16, True), "z": (20, True), "w": (5, False), "x": (6, False),
    "y": (2, False), "Y": (3, False), "U": (15, False), "P": (5, False), "X": (5, False),
}  # fmt: skip
_FCC = (fcc0, fcc1, fcc2, fcc3, fcc4, fcc5, fcc6, fcc7)  # noqa: F405
_FCSR = (fcsr0, fcsr1, fcsr2, fcsr3)  # noqa: F405


def _random_operands(template, rng):
    """Random operands that one template alternative accepts, following
    its letters (`jita.loongarch64.table`). None for the label and li
    forms, and for an am* draw whose rd repeats rk or rj."""
    ops: list[Any] = []
    for p in template[8:]:
        if p in "DJK":
            ops.append(gpr(rng.randrange(32)))  # noqa: F405
        elif p in "djka":
            ops.append(fpr(rng.randrange(32)))  # noqa: F405
        elif p in "CcE":
            ops.append(rng.choice(_FCC))
        elif p in "Ss":
            ops.append(rng.choice(_FCSR))
        elif p in _IMM_FIELDS:
            bits, signed = _IMM_FIELDS[p]
            v = rng.getrandbits(bits)
            ops.append(v - (1 << bits) if signed and v >> (bits - 1) else v)
        elif p in "MN":
            bits = 5 if p == "M" else 6
            msb, lsb = sorted(rng.getrandbits(bits) for _ in range(2))[::-1]
            ops += [msb, lsb]
        elif p in "po":
            bits = 14 if p == "p" else 16
            ops.append((rng.getrandbits(bits) - (1 << (bits - 1))) * 4)
        elif p == "q":
            ops.append(rng.randrange(1, 5))
        elif p == "!":
            if ops[0].code and ops[0] in (ops[1], ops[2]):
                return None
        elif p != "=":
            return None  # labels and li are covered elsewhere
    return ops


@requires_loongarch64_oracle
def test_oracle_random_operands():
    # Random registers and immediates for every template alternative
    # without a label, a few per alternative, in one assembly.
    rng = random.Random(4242)
    lines, words = [], []
    for key, template in sorted(MAP_OP.items()):
        fn = MNEMONICS[key.rpartition("_")[0]]
        for alt in template.split("|"):
            for _ in range(4):
                ops = _random_operands(alt, rng)
                if ops is None:
                    continue
                lines.append(gnu_text(fn, ops))
                words.append(encode(fn, *ops)[0])
    code = assemble("\n".join(lines), arch="loongarch64")
    assert len(lines) > 1000
    bad = [f"{line}: GNU {code[4 * i : 4 * i + 4].hex()}, jita {w.hex()}" for i, (line, w) in enumerate(zip(lines, words))
           if code[4 * i : 4 * i + 4] != w]  # fmt: skip
    assert not bad, "\n".join(bad[:20])
