"""riscv64 instruction templates.

Written for jita from the RISC-V unprivileged specification and checked
against GNU as; DynASM's RISC-V port is not used for encodings. The
template syntax follows `jita.aarch64.table`.

Keys are "mnemonic_argcount". A template is a "|" separated list of
alternatives, the first one that accepts the operands wins. An
alternative starts with the 32 bit instruction word in 8 hex digits, with
every operand field zero, followed by one character per operand action:

  D N M     integer register into rd (bit 7), rs1 (15), rs2 (20)
  d n m a   floating point register into rd, rs1, rs2, rs3 (27)
  E         rd again into rs1 (no operand)
  ^         rs1 again into rs2 (no operand)
  I         signed 12 bit immediate at bit 20 (I-type)
  U         20 bit immediate at bit 12 (lui, auipc)
  H W       shift amount at bit 20, 6 bits (RV64) or 5 bits (the *w forms)
  C         CSR number or name at bit 20
  K         unsigned 5 bit immediate at bit 15 (csrrwi and friends)
  L S       memory operand: base into rs1, 12 bit offset I-type or S-type
  A         memory operand with offset 0 (atomics): base into rs1
  B J       label, B-type (+-4KB) or J-type (+-1MB) offset
  G         label or Extern reached through auipc: the word is emitted
            after `auipc rs1, hi20` and gets lo12 (PCREL32); rs1 must not
            be zero
  g         G that allows zero (`lla zero, lbl` is accepted by GNU as)
  ~         rs1 must differ from rs2 (a store to a label: auipc into the
            register being stored would overwrite the value)
  P Q       fence predecessor (bit 24) and successor (bit 20) sets
  #         any 64 bit value, expanded by `li`
  r         rounding mode from the `rm=` keyword at bit 12 (no operand)

Floating point instructions with an `r` take `rm=` and default to `dyn`,
as GNU as does. The conversions that are always exact (`fcvt.d.w`,
`fcvt.d.wu`, `fcvt.d.s`) have rm 0 in the word and no keyword, also as in
GNU as.
"""

# Rounding mode names of the `rm=` keyword.
MAP_RM: dict[str, int] = {"rne": 0, "rtz": 1, "rdn": 2, "rup": 3, "rmm": 4, "dyn": 7}

# CSR names accepted in place of a number.
MAP_CSR: dict[str, int] = {"fflags": 0x001, "frm": 0x002, "fcsr": 0x003, "cycle": 0xC00, "time": 0xC01, "instret": 0xC02}

# fence sets: the bits of i, o, r, w. GNU as wants them in this order.
MAP_FENCE: dict[str, int] = {"i": 8, "o": 4, "r": 2, "w": 1}

MAP_OP: dict[str, str] = {
    # RV64I: upper immediates, jumps and branches.
    "lui_2": "00000037DU",
    "auipc_2": "00000017DU",
    "jal_1": "000000efJ",
    "jal_2": "0000006fDJ",
    "jalr_1": "000000e7N",
    "jalr_2": "00000067DN|00000067DL|000000e7NI",
    "jalr_3": "00000067DNI",
    "beq_3": "00000063NMB",
    "bne_3": "00001063NMB",
    "blt_3": "00004063NMB",
    "bge_3": "00005063NMB",
    "bltu_3": "00006063NMB",
    "bgeu_3": "00007063NMB",

    # Loads and stores. With a label instead of a memory operand, a load
    # is `auipc rd; l* rd, lo(rd)` and a store takes a temporary register
    # for the auipc, not the one stored: `sd a0, lbl, t0`.
    "lb_2": "00000003DL|00000003DEG",
    "lh_2": "00001003DL|00001003DEG",
    "lw_2": "00002003DL|00002003DEG",
    "ld_2": "00003003DL|00003003DEG",
    "lbu_2": "00004003DL|00004003DEG",
    "lhu_2": "00005003DL|00005003DEG",
    "lwu_2": "00006003DL|00006003DEG",
    "sb_2": "00000023MS",
    "sb_3": "00000023MGN~",
    "sh_2": "00001023MS",
    "sh_3": "00001023MGN~",
    "sw_2": "00002023MS",
    "sw_3": "00002023MGN~",
    "sd_2": "00003023MS",
    "sd_3": "00003023MGN~",

    # Integer arithmetic and logic.
    "addi_3": "00000013DNI",
    "slti_3": "00002013DNI",
    "sltiu_3": "00003013DNI",
    "xori_3": "00004013DNI",
    "ori_3": "00006013DNI",
    "andi_3": "00007013DNI",
    "slli_3": "00001013DNH",
    "srli_3": "00005013DNH",
    "srai_3": "40005013DNH",
    "add_3": "00000033DNM",
    "sub_3": "40000033DNM",
    "sll_3": "00001033DNM",
    "slt_3": "00002033DNM",
    "sltu_3": "00003033DNM",
    "xor_3": "00004033DNM",
    "srl_3": "00005033DNM",
    "sra_3": "40005033DNM",
    "or_3": "00006033DNM",
    "and_3": "00007033DNM",
    "addiw_3": "0000001bDNI",
    "slliw_3": "0000101bDNW",
    "srliw_3": "0000501bDNW",
    "sraiw_3": "4000501bDNW",
    "addw_3": "0000003bDNM",
    "subw_3": "4000003bDNM",
    "sllw_3": "0000103bDNM",
    "srlw_3": "0000503bDNM",
    "sraw_3": "4000503bDNM",

    # System.
    "ecall_0": "00000073",
    "ebreak_0": "00100073",
    "fence_0": "0ff0000f",
    "fence_2": "0000000fPQ",
    "fence.tso_0": "8330000f",
    "fence.i_0": "0000100f",

    # Zicsr.
    "csrrw_3": "00001073DCN",
    "csrrs_3": "00002073DCN",
    "csrrc_3": "00003073DCN",
    "csrrwi_3": "00005073DCK",
    "csrrsi_3": "00006073DCK",
    "csrrci_3": "00007073DCK",

    # M.
    "mul_3": "02000033DNM",
    "mulh_3": "02001033DNM",
    "mulhsu_3": "02002033DNM",
    "mulhu_3": "02003033DNM",
    "div_3": "02004033DNM",
    "divu_3": "02005033DNM",
    "rem_3": "02006033DNM",
    "remu_3": "02007033DNM",
    "mulw_3": "0200003bDNM",
    "divw_3": "0200403bDNM",
    "divuw_3": "0200503bDNM",
    "remw_3": "0200603bDNM",
    "remuw_3": "0200703bDNM",

    # F and D. Loads and stores with a label take a temporary integer
    # register for the auipc: `fld fa0, lbl, t0`.
    "flw_2": "00002007dL",
    "flw_3": "00002007dGN",
    "fsw_2": "00002027mS",
    "fsw_3": "00002027mGN",
    "fld_2": "00003007dL",
    "fld_3": "00003007dGN",
    "fsd_2": "00003027mS",
    "fsd_3": "00003027mGN",
    "fadd.s_3": "00000053dnmr",
    "fsub.s_3": "08000053dnmr",
    "fmul.s_3": "10000053dnmr",
    "fdiv.s_3": "18000053dnmr",
    "fsqrt.s_2": "58000053dnr",
    "fsgnj.s_3": "20000053dnm",
    "fsgnjn.s_3": "20001053dnm",
    "fsgnjx.s_3": "20002053dnm",
    "fmin.s_3": "28000053dnm",
    "fmax.s_3": "28001053dnm",
    "fmadd.s_4": "00000043dnmar",
    "fmsub.s_4": "00000047dnmar",
    "fnmsub.s_4": "0000004bdnmar",
    "fnmadd.s_4": "0000004fdnmar",
    "fadd.d_3": "02000053dnmr",
    "fsub.d_3": "0a000053dnmr",
    "fmul.d_3": "12000053dnmr",
    "fdiv.d_3": "1a000053dnmr",
    "fsqrt.d_2": "5a000053dnr",
    "fsgnj.d_3": "22000053dnm",
    "fsgnjn.d_3": "22001053dnm",
    "fsgnjx.d_3": "22002053dnm",
    "fmin.d_3": "2a000053dnm",
    "fmax.d_3": "2a001053dnm",
    "fmadd.d_4": "02000043dnmar",
    "fmsub.d_4": "02000047dnmar",
    "fnmsub.d_4": "0200004bdnmar",
    "fnmadd.d_4": "0200004fdnmar",
    "feq.s_3": "a0002053Dnm",
    "flt.s_3": "a0001053Dnm",
    "fle.s_3": "a0000053Dnm",
    "feq.d_3": "a2002053Dnm",
    "flt.d_3": "a2001053Dnm",
    "fle.d_3": "a2000053Dnm",
    "fclass.s_2": "e0001053Dn",
    "fclass.d_2": "e2001053Dn",
    "fcvt.w.s_2": "c0000053Dnr",
    "fcvt.wu.s_2": "c0100053Dnr",
    "fcvt.l.s_2": "c0200053Dnr",
    "fcvt.lu.s_2": "c0300053Dnr",
    "fcvt.s.w_2": "d0000053dNr",
    "fcvt.s.wu_2": "d0100053dNr",
    "fcvt.s.l_2": "d0200053dNr",
    "fcvt.s.lu_2": "d0300053dNr",
    "fcvt.w.d_2": "c2000053Dnr",
    "fcvt.wu.d_2": "c2100053Dnr",
    "fcvt.l.d_2": "c2200053Dnr",
    "fcvt.lu.d_2": "c2300053Dnr",
    "fcvt.d.w_2": "d2000053dN",
    "fcvt.d.wu_2": "d2100053dN",
    "fcvt.d.l_2": "d2200053dNr",
    "fcvt.d.lu_2": "d2300053dNr",
    "fcvt.s.d_2": "40100053dnr",
    "fcvt.d.s_2": "42000053dn",
    "fmv.x.w_2": "e0000053Dn",
    "fmv.w.x_2": "f0000053dN",
    "fmv.x.d_2": "e2000053Dn",
    "fmv.d.x_2": "f2000053dN",

    # Zba.
    "sh1add_3": "20002033DNM",
    "sh2add_3": "20004033DNM",
    "sh3add_3": "20006033DNM",
    "add.uw_3": "0800003bDNM",
    "sh1add.uw_3": "2000203bDNM",
    "sh2add.uw_3": "2000403bDNM",
    "sh3add.uw_3": "2000603bDNM",
    "slli.uw_3": "0800101bDNH",

    # Zbb.
    "andn_3": "40007033DNM",
    "orn_3": "40006033DNM",
    "xnor_3": "40004033DNM",
    "clz_2": "60001013DN",
    "ctz_2": "60101013DN",
    "cpop_2": "60201013DN",
    "clzw_2": "6000101bDN",
    "ctzw_2": "6010101bDN",
    "cpopw_2": "6020101bDN",
    "max_3": "0a006033DNM",
    "maxu_3": "0a007033DNM",
    "min_3": "0a004033DNM",
    "minu_3": "0a005033DNM",
    "sext.b_2": "60401013DN",
    "sext.h_2": "60501013DN",
    "zext.h_2": "0800403bDN",
    "rol_3": "60001033DNM",
    "rolw_3": "6000103bDNM",
    "ror_3": "60005033DNM",
    "rorw_3": "6000503bDNM",
    "rori_3": "60005013DNH",
    "roriw_3": "6000501bDNW",
    "orc.b_2": "28705013DN",
    "rev8_2": "6b805013DN",

    # Single instruction pseudo instructions (GNU as spellings).
    "nop_0": "00000013",
    "mv_2": "00000013DN",
    "not_2": "fff04013DN",
    "neg_2": "40000033DM",
    "negw_2": "4000003bDM",
    "sext.w_2": "0000001bDN",
    "zext.b_2": "0ff07013DN",
    "zext.w_2": "0800003bDN",
    "seqz_2": "00103013DN",
    "snez_2": "00003033DM",
    "sltz_2": "00002033DN",
    "sgtz_2": "00002033DM",
    "sgt_3": "00002033DMN",
    "sgtu_3": "00003033DMN",
    "beqz_2": "00000063NB",
    "bnez_2": "00001063NB",
    "blez_2": "00005063MB",
    "bgez_2": "00005063NB",
    "bltz_2": "00004063NB",
    "bgtz_2": "00004063MB",
    "bgt_3": "00004063MNB",
    "ble_3": "00005063MNB",
    "bgtu_3": "00006063MNB",
    "bleu_3": "00007063MNB",
    "j_1": "0000006fJ",
    "jr_1": "00000067N",
    "jr_2": "00000067NI",
    "ret_0": "00008067",
    "fmv.s_2": "20000053dn^",
    "fneg.s_2": "20001053dn^",
    "fabs.s_2": "20002053dn^",
    "fmv.d_2": "22000053dn^",
    "fneg.d_2": "22001053dn^",
    "fabs.d_2": "22002053dn^",
    "fgt.s_3": "a0001053Dmn",
    "fge.s_3": "a0000053Dmn",
    "fgt.d_3": "a2001053Dmn",
    "fge.d_3": "a2000053Dmn",
    "fmv.x.s_2": "e0000053Dn",
    "fmv.s.x_2": "f0000053dN",
    "unimp_0": "c0001073",
    "csrr_2": "00002073DC",
    "csrw_2": "00001073CN",
    "csrs_2": "00002073CN",
    "csrc_2": "00003073CN",
    "csrwi_2": "00005073CK",
    "csrsi_2": "00006073CK",
    "csrci_2": "00007073CK",
    "frcsr_1": "00302073D",
    "fscsr_1": "00301073N",
    "fscsr_2": "00301073DN",
    "frrm_1": "00202073D",
    "fsrm_1": "00201073N",
    "fsrm_2": "00201073DN",
    "fsrmi_1": "00205073K",
    "fsrmi_2": "00205073DK",
    "frflags_1": "00102073D",
    "fsflags_1": "00101073N",
    "fsflags_2": "00101073DN",
    "fsflagsi_1": "00105073K",
    "fsflagsi_2": "00105073DK",
    "rdcycle_1": "c0002073D",
    "rdtime_1": "c0102073D",
    "rdinstret_1": "c0202073D",

    # Multi instruction pseudo instructions.
    "li_2": "00000013D#",
    "lla_2": "00000013DEg",
    "la_2": "00000013DEG",
    "call_1": "000080e7G",
    "tail_1": "00030067G",
}  # fmt: skip


def _atomics() -> None:
    # lr, sc and the AMOs: funct5 at bit 27, width in funct3, and the
    # .aq, .rl and .aqrl orderings at bits 26 and 25.
    funct5 = {
        "lr": 0x02, "sc": 0x03, "amoswap": 0x01, "amoadd": 0x00, "amoxor": 0x04,
        "amoand": 0x0C, "amoor": 0x08, "amomin": 0x10, "amomax": 0x14,
        "amominu": 0x18, "amomaxu": 0x1C,
    }  # fmt: skip
    for name, f5 in funct5.items():
        for width, f3 in (("w", 2), ("d", 3)):
            for order, bits in (("", 0), (".aq", 2), (".rl", 1), (".aqrl", 3)):
                word = f5 << 27 | bits << 25 | f3 << 12 | 0x2F
                if name == "lr":
                    MAP_OP[f"{name}.{width}{order}_2"] = f"{word:08x}DA"
                else:
                    MAP_OP[f"{name}.{width}{order}_3"] = f"{word:08x}DMA"


_atomics()
