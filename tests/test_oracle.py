"""Self-tests for the oracle harness in tests/oracle.py.

These do not test jita itself; they check that assemble()/disassemble()/
normalize()/roundtrip() agree with each other and with GNU binutils, so
encoder tests can rely on them.
"""

import pytest

from oracle import OracleError, assemble, disassemble, normalize, requires_oracle, roundtrip


@requires_oracle
def test_assemble_mov_load():
    code = assemble("mov rax, qword ptr [rbx+8]")
    assert code == bytes.fromhex("488b4308")


@requires_oracle
def test_disassemble_mov_load():
    code = assemble("mov rax, qword ptr [rbx+8]")
    assert disassemble(code) == ["mov rax,qword ptr [rbx+0x8]"]


@requires_oracle
def test_roundtrip_mov_load():
    text = "mov rax, qword ptr [rbx+8]"
    roundtrip(text, assemble(text))


@requires_oracle
def test_disassemble_add_imm():
    code = assemble("add eax, 1")
    assert disassemble(code) == ["add eax,0x1"]


@requires_oracle
def test_roundtrip_add_imm():
    text = "add eax, 1"
    roundtrip(text, assemble(text))


@requires_oracle
def test_roundtrip_jmp_label():
    # A local label lets `as` resolve the short jump at assembly time
    # (unlike a bare numeric target, which needs a relocation that
    # `objcopy -O binary` drops). objdump then prints the resolved,
    # absolute-in-buffer target, "0x3" here: 2 bytes for the jmp itself,
    # then the label immediately follows the 1-byte nop.
    src = "jmp 1f\nnop\n1:"
    code = assemble(src)
    assert code == bytes.fromhex("eb0190")
    roundtrip("jmp 0x3\nnop", code)


@requires_oracle
def test_disassemble_sse():
    code = assemble("movaps xmm0, xmm1")
    assert disassemble(code) == ["movaps xmm0,xmm1"]


@requires_oracle
def test_roundtrip_sse():
    text = "movaps xmm0, xmm1"
    roundtrip(text, assemble(text))


@requires_oracle
def test_roundtrip_sse_mem():
    text = "movdqu xmm0, xmmword ptr [rax+16]"
    roundtrip(text, assemble(text))


@requires_oracle
def test_normalize_matches_disassemble_first_line():
    text = "add eax, 1"
    code = assemble(text)
    assert normalize(text) == disassemble(code)[0]


@requires_oracle
def test_normalize_hex_and_whitespace():
    assert normalize("MOV   EAX,  0X1A") == "mov eax,0x1a"
    assert normalize("add eax, 300") == "add eax,0x12c"
    assert normalize("lea rax, [rbx + rcx*4 + 16]") == "lea rax,[rbx+rcx*4+0x10]"


@requires_oracle
def test_assemble_caches_by_text():
    a = assemble("nop")
    b = assemble("nop")
    assert a == b == b"\x90"


@requires_oracle
def test_assemble_error_reports_as_stderr():
    with pytest.raises(OracleError) as excinfo:
        assemble("not_a_real_mnemonic_xyz eax, 1")
    assert "no such instruction" in str(excinfo.value)


@requires_oracle
def test_roundtrip_mismatch_raises_with_diff():
    text = "add eax, 1"
    other_code = assemble("add eax, 2")
    with pytest.raises(AssertionError) as excinfo:
        roundtrip(text, other_code)
    message = str(excinfo.value)
    assert "add eax,0x1" in message
    assert "add eax,0x2" in message
