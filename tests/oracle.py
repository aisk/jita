"""Oracle for testing jita's x64 encoder against GNU binutils.

This module assembles Intel-syntax text with GNU ``as`` and disassembles raw
machine code with ``objdump``, so encoder tests can check that jita produces
the same bytes (or at least round-trips to the same instruction text) as the
reference toolchain.

Normalization rules (shared by `normalize()` and the text produced by
`disassemble()`, so both sides of a comparison go through the same pipeline):

- Strip everything from a ``#`` character onward (objdump comment, e.g. a
  resolved rip-relative target: ``mov rax,QWORD PTR [rip+0x8]  # 0xf``).
- Lowercase the whole line. This normalizes mnemonics, registers, size
  keywords such as ``QWORD PTR`` -> ``qword ptr``, and hex digits.
- Collapse any run of whitespace to a single space, then strip the ends.
- Remove whitespace around ``,``, ``*``, ``+`` and ``-`` (objdump never
  spaces any of these).
- Convert bare decimal integers (immediates and displacements, e.g. the `8`
  in `[rbx+8]` or `1` in `add eax, 1`) to lowercase `0x..` hex, matching
  objdump's convention of always rendering immediates/displacements in hex.
  A number is only converted when it is not already part of a `0x..` token
  and not a SIB scale factor (`reg*N`, which objdump always prints in
  decimal, e.g. `[rbx+rcx*4+0x10]`).

Known limitations:

- Negative immediates are sign/width dependent in objdump's output (e.g.
  `mov eax, -1` disassembles as `mov eax,0xffffffff`, not `mov eax,-0x1`,
  because objdump renders the full zero/sign-extended machine value).
  `normalize()` has no operand-width information and will only flip the
  sign of the literal, so round trips with negative immediates on operands
  narrower than 64 bits should use the pre-computed unsigned hex form, or
  expect `roundtrip()` to report a mismatch.
- A GAS Intel-syntax memory operand omits `PTR` when the size is inferable
  from a register operand (e.g. `mov rax, [rbx+8]`), but objdump always
  prints it. Source text passed to `assemble()`/`roundtrip()` should include
  an explicit size (`qword ptr`, `dword ptr`, ...) whenever the operand size
  is not otherwise obvious, to match objdump's rendering.
- `roundtrip()` expects one instruction per source line, with any labels on
  their own line (a line matching `label:` is skipped rather than compared
  to an instruction); a label sharing a line with an instruction is not
  handled specially.
"""

import glob
import os
import platform
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

__all__ = [
    "OracleError",
    "assemble",
    "disassemble",
    "normalize",
    "roundtrip",
    "available",
    "requires_oracle",
    "aarch64_available",
    "aarch64_assembler",
    "requires_aarch64_oracle",
    "aarch64_disassemble",
    "requires_aarch64_disassembler",
]


class OracleError(RuntimeError):
    """Raised when the external assembler/disassembler pipeline fails."""




def _is_gnu(path: str) -> bool:
    try:
        out = subprocess.run([path, "--version"], capture_output=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return False
    return b"GNU" in out


def _x64_binutils() -> tuple[str, str, str]:
    """(as, objcopy, objdump) for x64: the native GNU tools on an x86 host,
    otherwise the x86_64-linux-gnu cross tools; empty strings if neither."""
    if platform.machine().lower() in ("x86_64", "amd64"):
        native = ("/usr/bin/as", "/usr/bin/objcopy", "/usr/bin/objdump")
        if all(os.access(t, os.X_OK) for t in native) and _is_gnu(native[0]):
            return native
    cross = [shutil.which(f"x86_64-linux-gnu-{n}") for n in ("as", "objcopy", "objdump")]
    if all(cross):
        return (cross[0] or "", cross[1] or "", cross[2] or "")
    return ("", "", "")


AS, OBJCOPY, OBJDUMP = _x64_binutils()

# text -> assembled bytes, so repeated calls with the same source (common in
# parametrized tests) don't keep shelling out to `as`/`objcopy`.
_assemble_cache: dict[str, bytes] = {}

# A disassembly line looks like "   0:\t48 8b 43 08          \tmov    rax,...".
# A continuation line for a long instruction has no third (mnemonic) field:
# "  1c:\t56 34 12 ".
_LINE_RE = re.compile(r"^\s*[0-9a-f]+:\t(.*)$")

# Bare decimal integer, not part of a "0x.." token (word-boundary logic keeps
# us out of hex literals: there is no \b between the "0"/"x" and the digits
# that follow) and not a SIB scale factor (the "4" in "rcx*4").
_DECIMAL_RE = re.compile(r"(?<!\*)\b\d+\b")

# A label-only line, e.g. "1:" or "loop_start:".
_LABEL_RE = re.compile(r"^[\w.$]+:$")


def available() -> bool:
    """Return True if GNU `as`, `objcopy` and `objdump` for x64 were found."""
    return bool(AS)


requires_oracle = pytest.mark.skipif(
    not available(), reason="as/objcopy/objdump not available"
)


def assemble(text: str, arch: str = "x64") -> bytes:
    """Assemble Intel-syntax x64 text and return the raw machine code bytes.

    With `arch="aarch64"`, assemble GNU syntax aarch64 text instead (see
    `aarch64_assembler()`).

    `text` is wrapped in `.intel_syntax noprefix` / `.text` and assembled
    with `as --64` in a temporary directory; the resulting object file is
    reduced to raw bytes with `objcopy -O binary`. `text` may contain
    multiple lines (multiple instructions, labels, etc).

    Results are cached by the exact `text` given. Raises `OracleError`
    (with `as`'s stderr as the message) if assembly fails.
    """
    if arch == "aarch64":
        return _assemble_aarch64(text)
    if arch != "x64":
        raise ValueError(f"unknown arch {arch!r}")
    if text in _assemble_cache:
        return _assemble_cache[text]

    source = ".intel_syntax noprefix\n.text\n" + text
    if not source.endswith("\n"):
        source += "\n"

    with tempfile.TemporaryDirectory(prefix="jita-oracle-") as tmp_dir:
        tmp = Path(tmp_dir)
        asm_path = tmp / "in.s"
        obj_path = tmp / "out.o"
        bin_path = tmp / "out.bin"
        asm_path.write_text(source)

        as_result = subprocess.run(
            [AS, "--64", "-o", str(obj_path), str(asm_path)],
            capture_output=True,
            text=True,
        )
        if as_result.returncode != 0:
            raise OracleError(as_result.stderr)

        objcopy_result = subprocess.run(
            [OBJCOPY, "-O", "binary", str(obj_path), str(bin_path)],
            capture_output=True,
            text=True,
        )
        if objcopy_result.returncode != 0:
            raise OracleError(objcopy_result.stderr)

        code = bin_path.read_bytes()

    _assemble_cache[text] = code
    return code


# aarch64: GNU cross binutils when installed, otherwise LLVM's integrated
# assembler (llvm-mc), which is often present as part of clang.


def _llvm_version(path: str) -> int:
    m = re.search(r"/llvm-(\d+)/", path)
    return int(m.group(1)) if m else -1


def _runs(path: str) -> bool:
    """True if `path --version` runs successfully."""
    try:
        return subprocess.run([path, "--version"], capture_output=True, timeout=30).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


_llvm_tools: dict[str, str | None] = {}


def _llvm_tool(name: str) -> str | None:
    """`name` on PATH, else the newest /usr/lib/llvm-N/bin/`name`, if it runs."""
    if name in _llvm_tools:
        return _llvm_tools[name]
    which = shutil.which(name)
    candidates = [which] if which else []
    candidates += sorted(glob.glob(f"/usr/lib/llvm-*/bin/{name}"), key=_llvm_version, reverse=True)
    found = next((p for p in candidates if os.access(p, os.X_OK) and _runs(p)), None)
    _llvm_tools[name] = found
    return found


def aarch64_assembler() -> tuple[str, list[str], str] | None:
    """(name, assembler command, objcopy) for aarch64, or None."""
    gas = shutil.which("aarch64-linux-gnu-as")
    gobjcopy = shutil.which("aarch64-linux-gnu-objcopy")
    if gas and gobjcopy:
        return ("gas", [gas, "-march=armv8.5-a"], gobjcopy)
    mc, objcopy = _llvm_tool("llvm-mc"), _llvm_tool("llvm-objcopy")
    if mc and objcopy:
        return ("llvm-mc", [mc, "-triple=aarch64", "-filetype=obj", "-mattr=+v8.5a"], objcopy)
    return None


def aarch64_available() -> bool:
    return aarch64_assembler() is not None


requires_aarch64_oracle = pytest.mark.skipif(
    not aarch64_available(), reason="no aarch64 assembler (aarch64-linux-gnu-as or llvm-mc)"
)

def aarch64_disassemble(code: bytes) -> list[str]:
    """Disassemble aarch64 machine code with llvm-mc, one line per word,
    whitespace collapsed (`add x0, x1, #1`). A word that does not decode
    gives `<invalid>`."""
    mc = _llvm_tool("llvm-mc")
    if mc is None:
        raise OracleError("llvm-mc not available")
    if len(code) % 4:
        raise ValueError("aarch64 code must be a multiple of 4 bytes")

    def run(chunk: bytes) -> list[str]:
        r = subprocess.run(
            [mc, "--disassemble", "-triple=aarch64", "-mattr=+v8.5a"],
            input=" ".join(f"0x{b:02x}" for b in chunk),
            capture_output=True,
            text=True,
        )
        if r.returncode != 0:
            return []
        return [
            re.sub(r"\s+", " ", ln.strip())
            for ln in r.stdout.splitlines()
            if ln.strip() and not ln.strip().startswith(".")
        ]

    lines = run(code)
    if len(lines) == len(code) // 4:
        return lines
    # Some word did not decode: go one word at a time to keep positions.
    return [(run(code[i : i + 4]) or ["<invalid>"])[0] for i in range(0, len(code), 4)]


requires_aarch64_disassembler = pytest.mark.skipif(
    _llvm_tool("llvm-mc") is None, reason="llvm-mc not available"
)

_aarch64_cache: dict[str, bytes] = {}


def _assemble_aarch64(text: str) -> bytes:
    if text in _aarch64_cache:
        return _aarch64_cache[text]
    tools = aarch64_assembler()
    if tools is None:
        raise OracleError("no aarch64 assembler available")
    _, as_cmd, objcopy = tools
    source = ".text\n" + text
    if not source.endswith("\n"):
        source += "\n"
    with tempfile.TemporaryDirectory(prefix="jita-oracle-") as tmp_dir:
        tmp = Path(tmp_dir)
        (tmp / "in.s").write_text(source)
        r = subprocess.run([*as_cmd, "-o", str(tmp / "out.o"), str(tmp / "in.s")], capture_output=True, text=True)
        if r.returncode != 0:
            raise OracleError(r.stderr)
        r = subprocess.run(
            [objcopy, "-O", "binary", "-j", ".text", str(tmp / "out.o"), str(tmp / "out.bin")],
            capture_output=True,
            text=True,
        )
        if r.returncode != 0:
            raise OracleError(r.stderr)
        code = (tmp / "out.bin").read_bytes()
    _aarch64_cache[text] = code
    return code


def disassemble(code: bytes) -> list[str]:
    """Disassemble raw x64 machine code and return one normalized line per
    instruction.

    Runs `objdump -D -b binary -m i386:x86-64 -M intel` on `code` and, for
    each decoded instruction, strips the address and hex-byte columns and
    normalizes the remaining text (see module docstring). Continuation lines
    that objdump emits for instructions whose encoding is too long for one
    line (address + hex bytes only, no mnemonic) are skipped, since the
    mnemonic and operands already appeared on the instruction's first line.
    """
    with tempfile.TemporaryDirectory(prefix="jita-oracle-") as tmp_dir:
        bin_path = Path(tmp_dir) / "in.bin"
        bin_path.write_bytes(code)

        result = subprocess.run(
            [
                OBJDUMP,
                "-D",
                "-b",
                "binary",
                "-m",
                "i386:x86-64",
                "-M",
                "intel",
                str(bin_path),
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise OracleError(result.stderr)
        output = result.stdout

    lines: list[str] = []
    for raw_line in output.splitlines():
        match = _LINE_RE.match(raw_line)
        if match is None:
            continue
        rest = match.group(1)
        if rest.strip() == "...":
            # objdump elides a run of identical (typically all-zero) bytes.
            continue
        parts = rest.split("\t")
        if len(parts) < 2:
            # Continuation line: only hex bytes, no mnemonic/operand field.
            continue
        normalized = _normalize_line(parts[1])
        if normalized:
            lines.append(normalized)
    return lines


def _dec_to_hex(match: re.Match[str]) -> str:
    return hex(int(match.group(0)))


def _normalize_line(text: str) -> str:
    text = text.split("#", 1)[0]
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s*,\s*", ",", text)
    text = re.sub(r"\s*\*\s*", "*", text)
    text = re.sub(r"\s*\+\s*", "+", text)
    text = re.sub(r"\s*-\s*", "-", text)
    text = _DECIMAL_RE.sub(_dec_to_hex, text)
    return text


def normalize(text: str) -> str:
    """Normalize a single Intel-syntax instruction string.

    Applies the same normalization as `disassemble()` (see module
    docstring), so that for a single-instruction round trip
    `normalize(src) == disassemble(assemble(src))[0]`.
    """
    return _normalize_line(text)


def roundtrip(text: str, code: bytes) -> None:
    """Assert that disassembling `code` yields exactly the normalized
    instruction lines of `text`.

    `text` may contain multiple lines; blank lines and label-only lines
    (`some_label:`) are skipped, and every remaining line is normalized and
    compared, in order, against `disassemble(code)`. On mismatch, raises
    `AssertionError` with a side-by-side diff of expected vs. actual lines
    and the hex of `code`.
    """
    expected: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or _LABEL_RE.match(line):
            continue
        expected.append(normalize(line))

    actual = disassemble(code)

    if actual != expected:
        width = max((len(e) for e in expected), default=0)
        rows = []
        for i in range(max(len(expected), len(actual))):
            e = expected[i] if i < len(expected) else "<missing>"
            a = actual[i] if i < len(actual) else "<missing>"
            marker = "==" if e == a else "!!"
            rows.append(f"  {marker} expected: {e!r:<{width + 2}} actual: {a!r}")
        diff = "\n".join(rows)
        raise AssertionError(
            f"roundtrip mismatch for text={text!r}\n"
            f"code hex: {code.hex()}\n{diff}"
        )
