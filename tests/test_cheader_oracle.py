"""jita.cheader layouts against a C compiler: every loaded struct and
union of the fixture headers is compiled into a program that prints its
sizeof and the offsetof of each field, and compared with ctypes."""

import ctypes
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("pycparser")
pytest.importorskip("pcpp")

from jita.cheader import Header, load  # noqa: E402

FIXTURES = Path(__file__).resolve().parent / "cheader"


def _compiler() -> str | None:
    for cc in (os.environ.get("CC"), "cc", "gcc", "clang"):
        if cc and (path := shutil.which(cc)):
            return path
    return None


def _other_target(cc: str | None) -> bool:
    """Whether `cc` builds for another architecture than this Python runs
    on, as with the host compiler under qemu-user."""
    if cc is None:
        return False
    try:
        target = subprocess.run([cc, "-dumpmachine"], capture_output=True, text=True, timeout=60).stdout
    except OSError:
        return False
    names = {"amd64": "x86_64", "arm64": "aarch64"}
    machine = platform.machine().lower()
    return bool(target) and names.get(machine, machine) not in target.replace("arm64", "aarch64")


CC = _compiler()
pytestmark = [
    pytest.mark.skipif(CC is None, reason="no C compiler (CC, cc, gcc or clang) on PATH"),
    pytest.mark.skipif(_other_target(CC), reason="the C compiler targets another architecture"),
]


def _c_name(name: str) -> str:
    for kind in ("struct", "union"):
        if name.startswith(kind + "_"):
            return f"{kind} {name[len(kind) + 1 :]}"
    return name


def _fields(ctype: type, prefix: str = "") -> list[tuple[str, int]]:
    """(C member designator, ctypes offset) of every field that is not a
    bit field, looking through anonymous members as C does."""
    anonymous = getattr(ctype, "_anonymous_", ())
    out = []
    for f in getattr(ctype, "_fields_"):
        name, t = f[0], f[1]
        if len(f) == 3:
            continue
        if name in anonymous:
            base = getattr(ctype, name).offset
            out += [(n, base + off) for n, off in _fields(t)]
            continue
        out.append((prefix + name, getattr(ctype, name).offset))
    return out


def _check(tmp_path: Path, header: str, h: Header) -> int:
    lines = [f'#include "{header}"', "#include <stdio.h>", "#include <stddef.h>", "int main(void) {"]
    expected = []
    for name, value in vars(h).items():
        if not (isinstance(value, type) and issubclass(value, (ctypes.Structure, ctypes.Union))):
            continue
        if not hasattr(value, "_fields_") or value.__name__ != name:
            continue
        c = _c_name(name)
        lines.append(f'printf("{name} %llu\\n", (unsigned long long)sizeof({c}));')
        expected.append(f"{name} {ctypes.sizeof(value)}")
        for member, offset in _fields(value):
            lines.append(f'printf("{name}.{member} %llu\\n", (unsigned long long)offsetof({c}, {member}));')
            expected.append(f"{name}.{member} {offset}")
    lines += ["return 0;", "}"]
    src = tmp_path / "oracle.c"
    src.write_text("\n".join(lines) + "\n")
    exe = tmp_path / ("oracle.exe" if sys.platform == "win32" else "oracle")
    assert CC is not None
    # MinGW's long double is the x87 type, ctypes' on Windows is MSVC's double.
    flags = ["-mlong-double-64"] if sys.platform == "win32" else []
    proc = subprocess.run(
        [CC, "-std=gnu11", *flags, "-DORACLE", f"-I{FIXTURES}", str(src), "-o", str(exe)],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    got = subprocess.run([str(exe)], check=True, capture_output=True, text=True).stdout.split("\n")
    assert got[:-1] == expected
    return len(expected)


@pytest.mark.parametrize("header", ["types.h", "review.h"])
def test_layouts_match_the_c_compiler(tmp_path, header):
    h = load(FIXTURES / header)
    assert _check(tmp_path, header, h) > 20
