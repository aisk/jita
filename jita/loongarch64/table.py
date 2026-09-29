"""loongarch64 instruction templates.

Written for jita from the LoongArch reference manual and checked against
GNU as (binutils' opcodes/loongarch-opc.c); the DynASM LoongArch port is
not used for encodings. The template syntax follows `jita.riscv64.table`.

Keys are "mnemonic_argcount". A template is a "|" separated list of
alternatives, the first one that accepts the operands wins. An
alternative starts with the 32 bit instruction word in 8 hex digits, with
every operand field zero, followed by one character per operand action:

  D J K     integer register into rd (bit 0), rj (5), rk (10)
  d j k a   floating point register into fd (0), fj (5), fk (10), fa (15)
  C c E     condition flag register into cd (0), cj (5), ca (15)
  S s       fcsr register at bit 0 or bit 5
  =         rd again into rj (no operand)
  i u       signed or unsigned 12 bit immediate at bit 10
  h         signed 16 bit immediate at bit 10 (addu16i.d)
  z         signed 20 bit immediate at bit 5
  w x       unsigned 5 or 6 bit immediate at bit 10 (shift amounts)
  M N       two operands msb, lsb of 5 or 6 bits at bits 16 and 10,
            msb >= lsb (bstrins, bstrpick)
  p o       byte offset, a multiple of 4, stored >> 2 at bit 10: signed
            14 bit (ll, sc, ldptr, stptr) or 16 bit (jirl)
  q         shift 1..4 stored minus 1 at bit 15 (alsl)
  y Y       unsigned 2 or 3 bit immediate at bit 15 (bytepick)
  U         unsigned 15 bit code at bit 0 (break, dbar, ...)
  P         unsigned 5 bit hint at bit 0 (preld)
  X         unsigned 5 bit immediate into rd and rj (ud)
  B R T     label, 16 bit (+-128KB), 21 bit (+-4MB) or 26 bit (+-128MB)
            branch offset in words
  V         label, 20 bit pc relative word offset at bit 5 (pcaddi)
  L         label or Extern reached through pcalau12i: the word is
            emitted after `pcalau12i rj, hi20` and gets the low 12 bits of
            the target at bit 10 (PCALA); rj must not be zero
  l         L that allows zero (`la.local $zero, lbl` is accepted by GNU as
            and leaves no register behind)
  O         label or Extern reached through pcaddu18i: the word is a jirl
            emitted after `pcaddu18i rj, hi20` (CALL36); rj must not be
            zero
  # %       any 64 bit value or any 32 bit value, expanded by li.d, li.w
  !         rd must differ from rk and rj unless it is zero (am*)
  ~         rj must differ from rd (a store to a label: pcalau12i into the
            register being stored would overwrite the value)
  m n       memory operand (`MemExpr`, built or from typed()) with base
            and offset: base into rj, offset as i (m) or as p (n)
  r         memory operand with base and index: base into rj, index
            into rk
  b         memory operand with a bare base, no offset: base into rj
            (am*, ldgt and friends)

A memory operand stands for the base and offset, or base and index,
operands of the same instruction, so `ld.w_2` takes `rd, mem` where
`ld.w_3` takes `rd, rj, si12`. The access width comes from the
mnemonic's last part (b, h, w, d, bu, hu, wu, du, s) and a typed()
operand of another size is rejected; preld and preldx have no width.
"""

MAP_OP: dict[str, str] = {
    # Integer arithmetic, logic and shifts.
    "add.w_3": "00100000DJK",
    "add.d_3": "00108000DJK",
    "sub.w_3": "00110000DJK",
    "sub.d_3": "00118000DJK",
    "slt_3": "00120000DJK",
    "sltu_3": "00128000DJK",
    "maskeqz_3": "00130000DJK",
    "masknez_3": "00138000DJK",
    "nor_3": "00140000DJK",
    "and_3": "00148000DJK",
    "or_3": "00150000DJK",
    "xor_3": "00158000DJK",
    "orn_3": "00160000DJK",
    "andn_3": "00168000DJK",
    "sll.w_3": "00170000DJK",
    "srl.w_3": "00178000DJK",
    "sra.w_3": "00180000DJK",
    "sll.d_3": "00188000DJK",
    "srl.d_3": "00190000DJK",
    "sra.d_3": "00198000DJK",
    "rotr.w_3": "001b0000DJK",
    "rotr.d_3": "001b8000DJK",
    "mul.w_3": "001c0000DJK",
    "mulh.w_3": "001c8000DJK",
    "mulh.wu_3": "001d0000DJK",
    "mul.d_3": "001d8000DJK",
    "mulh.d_3": "001e0000DJK",
    "mulh.du_3": "001e8000DJK",
    "mulw.d.w_3": "001f0000DJK",
    "mulw.d.wu_3": "001f8000DJK",
    "div.w_3": "00200000DJK",
    "mod.w_3": "00208000DJK",
    "div.wu_3": "00210000DJK",
    "mod.wu_3": "00218000DJK",
    "div.d_3": "00220000DJK",
    "mod.d_3": "00228000DJK",
    "div.du_3": "00230000DJK",
    "mod.du_3": "00238000DJK",
    "crc.w.b.w_3": "00240000DJK",
    "crc.w.h.w_3": "00248000DJK",
    "crc.w.w.w_3": "00250000DJK",
    "crc.w.d.w_3": "00258000DJK",
    "crcc.w.b.w_3": "00260000DJK",
    "crcc.w.h.w_3": "00268000DJK",
    "crcc.w.w.w_3": "00270000DJK",
    "crcc.w.d.w_3": "00278000DJK",
    "alsl.w_4": "00040000DJKq",
    "alsl.wu_4": "00060000DJKq",
    "alsl.d_4": "002c0000DJKq",
    "bytepick.w_4": "00080000DJKy",
    "bytepick.d_4": "000c0000DJKY",
    "slli.w_3": "00408000DJw",
    "slli.d_3": "00410000DJx",
    "srli.w_3": "00448000DJw",
    "srli.d_3": "00450000DJx",
    "srai.w_3": "00488000DJw",
    "srai.d_3": "00490000DJx",
    "rotri.w_3": "004c8000DJw",
    "rotri.d_3": "004d0000DJx",
    "bstrins.w_4": "00600000DJM",
    "bstrpick.w_4": "00608000DJM",
    "bstrins.d_4": "00800000DJN",
    "bstrpick.d_4": "00c00000DJN",
    "slti_3": "02000000DJi",
    "sltui_3": "02400000DJi",
    "addi.w_3": "02800000DJi",
    "addi.d_3": "02c00000DJi",
    "lu52i.d_3": "03000000DJi",
    "andi_3": "03400000DJu",
    "ori_3": "03800000DJu",
    "xori_3": "03c00000DJu",
    "addu16i.d_3": "10000000DJh",
    "lu12i.w_2": "14000000Dz",
    "lu32i.d_2": "16000000Dz",
    "pcaddi_2": "18000000Dz|18000000DV",
    "pcalau12i_2": "1a000000Dz",
    "pcaddu12i_2": "1c000000Dz",
    "pcaddu18i_2": "1e000000Dz",

    # Bit manipulation and counters.
    "clo.w_2": "00001000DJ",
    "clz.w_2": "00001400DJ",
    "cto.w_2": "00001800DJ",
    "ctz.w_2": "00001c00DJ",
    "clo.d_2": "00002000DJ",
    "clz.d_2": "00002400DJ",
    "cto.d_2": "00002800DJ",
    "ctz.d_2": "00002c00DJ",
    "revb.2h_2": "00003000DJ",
    "revb.4h_2": "00003400DJ",
    "revb.2w_2": "00003800DJ",
    "revb.d_2": "00003c00DJ",
    "revh.2w_2": "00004000DJ",
    "revh.d_2": "00004400DJ",
    "bitrev.4b_2": "00004800DJ",
    "bitrev.8b_2": "00004c00DJ",
    "bitrev.w_2": "00005000DJ",
    "bitrev.d_2": "00005400DJ",
    "ext.w.h_2": "00005800DJ",
    "ext.w.b_2": "00005c00DJ",
    "rdtimel.w_2": "00006000DJ",
    "rdtimeh.w_2": "00006400DJ",
    "rdtime.d_2": "00006800DJ",
    "cpucfg_2": "00006c00DJ",
    "asrtle.d_2": "00010000JK",
    "asrtgt.d_2": "00018000JK",

    # Misc.
    "break_1": "002a0000U",
    "dbcl_1": "002a8000U",
    "syscall_1": "002b0000U",
    "dbar_1": "38720000U",
    "ibar_1": "38728000U",

    # Loads and stores. With a label, a load is `pcalau12i rd; ld.* rd,
    # rd, lo12` and a store or floating point access names the register
    # for the pcalau12i, not the one stored: `st.d $a0, lbl, $t0`.
    "ld.b_3": "28000000DJi",
    "ld.h_3": "28400000DJi",
    "ld.w_3": "28800000DJi",
    "ld.d_3": "28c00000DJi",
    "ld.bu_3": "2a000000DJi",
    "ld.hu_3": "2a400000DJi",
    "ld.wu_3": "2a800000DJi",
    "ld.b_2": "28000000D=L",
    "ld.h_2": "28400000D=L",
    "ld.w_2": "28800000D=L",
    "ld.d_2": "28c00000D=L",
    "ld.bu_2": "2a000000D=L",
    "ld.hu_2": "2a400000D=L",
    "ld.wu_2": "2a800000D=L",
    "st.b_3": "29000000DJi|29000000DLJ~",
    "st.h_3": "29400000DJi|29400000DLJ~",
    "st.w_3": "29800000DJi|29800000DLJ~",
    "st.d_3": "29c00000DJi|29c00000DLJ~",
    "preld_3": "2ac00000PJi",
    "ldx.b_3": "38000000DJK",
    "ldx.h_3": "38040000DJK",
    "ldx.w_3": "38080000DJK",
    "ldx.d_3": "380c0000DJK",
    "stx.b_3": "38100000DJK",
    "stx.h_3": "38140000DJK",
    "stx.w_3": "38180000DJK",
    "stx.d_3": "381c0000DJK",
    "ldx.bu_3": "38200000DJK",
    "ldx.hu_3": "38240000DJK",
    "ldx.wu_3": "38280000DJK",
    "preldx_3": "382c0000PJK",
    "ldptr.w_3": "24000000DJp",
    "stptr.w_3": "25000000DJp",
    "ldptr.d_3": "26000000DJp",
    "stptr.d_3": "27000000DJp",
    "ldgt.b_3": "38780000DJK",
    "ldgt.h_3": "38788000DJK",
    "ldgt.w_3": "38790000DJK",
    "ldgt.d_3": "38798000DJK",
    "ldle.b_3": "387a0000DJK",
    "ldle.h_3": "387a8000DJK",
    "ldle.w_3": "387b0000DJK",
    "ldle.d_3": "387b8000DJK",
    "stgt.b_3": "387c0000DJK",
    "stgt.h_3": "387c8000DJK",
    "stgt.w_3": "387d0000DJK",
    "stgt.d_3": "387d8000DJK",
    "stle.b_3": "387e0000DJK",
    "stle.h_3": "387e8000DJK",
    "stle.w_3": "387f0000DJK",
    "stle.d_3": "387f8000DJK",

    # Atomics: ll and sc take a byte offset, a multiple of 4.
    "ll.w_3": "20000000DJp",
    "sc.w_3": "21000000DJp",
    "ll.d_3": "22000000DJp",
    "sc.d_3": "23000000DJp",

    # Branches and jumps.
    "beqz_2": "40000000JR",
    "bnez_2": "44000000JR",
    "bceqz_2": "48000000cR",
    "bcnez_2": "48000100cR",
    "jirl_3": "4c000000DJo",
    "b_1": "50000000T",
    "bl_1": "54000000T",
    "beq_3": "58000000JDB",
    "bne_3": "5c000000JDB",
    "blt_3": "60000000JDB",
    "bge_3": "64000000JDB",
    "bltu_3": "68000000JDB",
    "bgeu_3": "6c000000JDB",

    # Floating point arithmetic.
    "fadd.s_3": "01008000djk",
    "fsub.s_3": "01028000djk",
    "fmul.s_3": "01048000djk",
    "fdiv.s_3": "01068000djk",
    "fmax.s_3": "01088000djk",
    "fmin.s_3": "010a8000djk",
    "fmaxa.s_3": "010c8000djk",
    "fmina.s_3": "010e8000djk",
    "fscaleb.s_3": "01108000djk",
    "fcopysign.s_3": "01128000djk",
    "fadd.d_3": "01010000djk",
    "fsub.d_3": "01030000djk",
    "fmul.d_3": "01050000djk",
    "fdiv.d_3": "01070000djk",
    "fmax.d_3": "01090000djk",
    "fmin.d_3": "010b0000djk",
    "fmaxa.d_3": "010d0000djk",
    "fmina.d_3": "010f0000djk",
    "fscaleb.d_3": "01110000djk",
    "fcopysign.d_3": "01130000djk",
    "fabs.s_2": "01140400dj",
    "fneg.s_2": "01141400dj",
    "flogb.s_2": "01142400dj",
    "fclass.s_2": "01143400dj",
    "fsqrt.s_2": "01144400dj",
    "frecip.s_2": "01145400dj",
    "frsqrt.s_2": "01146400dj",
    "fmov.s_2": "01149400dj",
    "fabs.d_2": "01140800dj",
    "fneg.d_2": "01141800dj",
    "flogb.d_2": "01142800dj",
    "fclass.d_2": "01143800dj",
    "fsqrt.d_2": "01144800dj",
    "frecip.d_2": "01145800dj",
    "frsqrt.d_2": "01146800dj",
    "fmov.d_2": "01149800dj",
    "fmadd.s_4": "08100000djka",
    "fmsub.s_4": "08500000djka",
    "fnmadd.s_4": "08900000djka",
    "fnmsub.s_4": "08d00000djka",
    "fmadd.d_4": "08200000djka",
    "fmsub.d_4": "08600000djka",
    "fnmadd.d_4": "08a00000djka",
    "fnmsub.d_4": "08e00000djka",
    "fsel_4": "0d000000djkE",

    # Floating point moves and conversions.
    "movgr2fr.w_2": "0114a400dJ",
    "movgr2fr.d_2": "0114a800dJ",
    "movgr2frh.w_2": "0114ac00dJ",
    "movfr2gr.s_2": "0114b400Dj",
    "movfr2gr.d_2": "0114b800Dj",
    "movfrh2gr.s_2": "0114bc00Dj",
    "movgr2fcsr_2": "0114c000SJ",
    "movfcsr2gr_2": "0114c800Ds",
    "movfr2cf_2": "0114d000Cj",
    "movcf2fr_2": "0114d400dc",
    "movgr2cf_2": "0114d800CJ",
    "movcf2gr_2": "0114dc00Dc",
    "fcvt.s.d_2": "01191800dj",
    "fcvt.d.s_2": "01192400dj",
    "ftintrm.w.s_2": "011a0400dj",
    "ftintrm.w.d_2": "011a0800dj",
    "ftintrm.l.s_2": "011a2400dj",
    "ftintrm.l.d_2": "011a2800dj",
    "ftintrp.w.s_2": "011a4400dj",
    "ftintrp.w.d_2": "011a4800dj",
    "ftintrp.l.s_2": "011a6400dj",
    "ftintrp.l.d_2": "011a6800dj",
    "ftintrz.w.s_2": "011a8400dj",
    "ftintrz.w.d_2": "011a8800dj",
    "ftintrz.l.s_2": "011aa400dj",
    "ftintrz.l.d_2": "011aa800dj",
    "ftintrne.w.s_2": "011ac400dj",
    "ftintrne.w.d_2": "011ac800dj",
    "ftintrne.l.s_2": "011ae400dj",
    "ftintrne.l.d_2": "011ae800dj",
    "ftint.w.s_2": "011b0400dj",
    "ftint.w.d_2": "011b0800dj",
    "ftint.l.s_2": "011b2400dj",
    "ftint.l.d_2": "011b2800dj",
    "ffint.s.w_2": "011d1000dj",
    "ffint.s.l_2": "011d1800dj",
    "ffint.d.w_2": "011d2000dj",
    "ffint.d.l_2": "011d2800dj",
    "frint.s_2": "011e4400dj",
    "frint.d_2": "011e4800dj",

    # Floating point loads and stores; with a label as for the integer
    # stores: `fld.d $fa0, lbl, $t0`.
    "fld.s_3": "2b000000dJi|2b000000dLJ",
    "fst.s_3": "2b400000dJi|2b400000dLJ",
    "fld.d_3": "2b800000dJi|2b800000dLJ",
    "fst.d_3": "2bc00000dJi|2bc00000dLJ",
    "fldx.s_3": "38300000dJK",
    "fldx.d_3": "38340000dJK",
    "fstx.s_3": "38380000dJK",
    "fstx.d_3": "383c0000dJK",
    "fldgt.s_3": "38740000dJK",
    "fldgt.d_3": "38748000dJK",
    "fldle.s_3": "38750000dJK",
    "fldle.d_3": "38758000dJK",
    "fstgt.s_3": "38760000dJK",
    "fstgt.d_3": "38768000dJK",
    "fstle.s_3": "38770000dJK",
    "fstle.d_3": "38778000dJK",

    # Single instruction aliases (GNU as spellings).
    "nop_0": "03400000",
    "move_2": "00150000DJ",
    "ret_0": "4c000020",
    "jr_1": "4c000000J",
    "ud_1": "38600400X",
    "rdcntvl.w_1": "00006000D",
    "rdcntvh.w_1": "00006400D",
    "rdcntid.w_1": "00006000J",
    "bgt_3": "60000000DJB",
    "ble_3": "64000000DJB",
    "bgtu_3": "68000000DJB",
    "bleu_3": "6c000000DJB",
    "bltz_2": "60000000JB",
    "bgez_2": "64000000JB",
    "bgtz_2": "60000000DB",
    "blez_2": "64000000DB",

    # Multi instruction pseudo instructions.
    "li.w_2": "00000000D%",
    "li.d_2": "00000000D#",
    "la.local_2": "02c00000D=l",
    "la.pcrel_2": "02c00000D=l",
    "call36_1": "4c000021O",
    "tail36_2": "4c000000JO",
    "call_1": "4c000021O",
    "tail_2": "4c000000JO",
}  # fmt: skip

# fcmp conditions: the cond field at bit 15.
MAP_FCMP: dict[str, int] = {
    "caf": 0x00, "saf": 0x01, "clt": 0x02, "slt": 0x03, "ceq": 0x04, "seq": 0x05,
    "cle": 0x06, "sle": 0x07, "cun": 0x08, "sun": 0x09, "cult": 0x0A, "sult": 0x0B,
    "cueq": 0x0C, "sueq": 0x0D, "cule": 0x0E, "sule": 0x0F, "cne": 0x10, "sne": 0x11,
    "cor": 0x14, "sor": 0x15, "cune": 0x18, "sune": 0x19,
}  # fmt: skip

# The conditions GNU as also accepts with the operands swapped.
MAP_FCMP_SWAPPED: dict[str, str] = {"sgt": "slt", "sge": "sle", "cugt": "cult", "cuge": "cule"}


def _generated() -> None:
    # fcmp.<cond>.s and .d write a condition flag register.
    for width, base in (("s", 0x0C100000), ("d", 0x0C200000)):
        for cond, code in MAP_FCMP.items():
            MAP_OP[f"fcmp.{cond}.{width}_3"] = f"{base | code << 15:08x}Cjk"
        for cond, same in MAP_FCMP_SWAPPED.items():
            MAP_OP[f"fcmp.{cond}.{width}_3"] = f"{base | MAP_FCMP[same] << 15:08x}Ckj"
    # am* and am*_db: rd, rk, rj. The _db forms are 0x90000 above the
    # plain ones. rd must differ from rk and rj (GNU as leaves amswap.w
    # alone, whose rd == rj form is `ud`).
    ops = ["amswap", "amadd", "amand", "amor", "amxor", "ammax", "ammin", "ammax", "ammin"]
    widths = [("w", "d")] * 7 + [("wu", "du")] * 2
    for i, (name, (w, d)) in enumerate(zip(ops, widths)):
        for db in ("", "_db"):
            for width, half in ((w, 0), (d, 1)):
                word = 0x38600000 + i * 0x10000 + half * 0x8000 + (0x90000 if db else 0)
                check = "" if name == "amswap" and db == "" and width == "w" else "!"
                MAP_OP[f"{name}{db}.{width}_3"] = f"{word:08x}DKJ{check}"
    _memory()


def _memory() -> None:
    # The memory operand forms of the loads, stores, prefetches and
    # atomics, next to their register forms (see m, n, r, b above).
    for key, tpl in list(MAP_OP.items()):
        name = key.rpartition("_")[0]
        head = name.split(".")[0]
        first = tpl.split("|")[0]
        word, ops = first[:8], first[8:]
        if head in ("ld", "st", "fld", "fst", "preld") and ops in ("DJi", "dJi", "PJi"):
            form = f"{word}{ops[0]}m"
        elif head in ("ldx", "stx", "fldx", "fstx", "preldx"):
            form = f"{word}{ops[0]}r"
        elif head in ("ldptr", "stptr", "ll", "sc"):
            form = f"{word}{ops[0]}n"
        elif head in ("ldgt", "ldle", "stgt", "stle", "fldgt", "fldle", "fstgt", "fstle"):
            MAP_OP[key] = f"{tpl}|{word}{ops[0]}bK"
            continue
        elif head.startswith("am") and ops.startswith("DKJ"):
            MAP_OP[key] = f"{tpl}|{word}DKb{ops[3:]}"
            continue
        else:
            continue
        two = f"{name}_2"
        MAP_OP[two] = f"{MAP_OP[two]}|{form}" if two in MAP_OP else form


_generated()
