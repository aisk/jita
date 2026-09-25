"""Run every example as a subprocess and check what it prints."""

import platform
import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"

EXPECTED = {
    "sum_array.py": ["sum(1..100) = 5050"],
    "dispatch_table.py": ["result = -68"],
    "call_extern.py": ["strlen via register: 15", "strlen via slot: 15"],
    "sse_dot.py": ["dot = 220.0"],
    "fragments.py": ["poly(10) + 6 + 300 = 3163", "jnz.short .L3", "add r10, qword ptr [rcx+r8*8-8]"],
    "conditional_codegen.py": [
        "reduce(op='add') = 31",
        "reduce(op='add', clamp=20, unroll=2) = 20",
        "reduce(op='max', unroll=2) = 12",
        "jnz.short .L1",
        "; rel8 -> .L2",
    ],
}

pytestmark = pytest.mark.skipif(
    platform.machine().lower() not in ("x86_64", "amd64") or sys.platform == "win32",
    reason="examples execute x64 code on a POSIX host",
)


def test_every_example_is_covered():
    assert sorted(p.name for p in EXAMPLES.glob("*.py")) == sorted(EXPECTED)


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_example(name):
    proc = subprocess.run(
        [sys.executable, str(EXAMPLES / name)], capture_output=True, text=True, timeout=60
    )
    assert proc.returncode == 0, proc.stderr
    lines = proc.stdout.splitlines()
    for want in EXPECTED[name]:
        assert any(want in line for line in lines), f"{want!r} not in output of {name}:\n{proc.stdout}"
