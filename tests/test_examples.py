"""Run every example and README code block as a subprocess and check them."""

import platform
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = ROOT / "examples"

EXPECTED = {
    "aarch64_hello.py": ["b.ne loop", "; rel19 -> loop", "ldr x3, [x0], #8"],
    "sum_array.py": ["sum(1..100) = 5050"],
    "dispatch_table.py": ["result = -68"],
    "call_extern.py": [
        "strlen via register: 15",
        "strlen via extern slot: 15",
        "strlen via slot: 15",
    ],
    "sse_dot.py": ["dot = 220.0"],
    "typed_struct.py": [
        "movsxd r9, dword ptr [rdi+rcx*4+24]",
        "totals = [3, 108, 0], flags = [1, 1, 1]",
        "sum = 111",
    ],
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


DOC_SNIPPETS = [
    (f"{doc.stem}{i}", code)
    for doc in (ROOT / "README.md", ROOT / "docs" / "reference.md")
    for i, code in enumerate(re.findall(r"```python\n(.*?)```", doc.read_text(), re.S))
]


@pytest.mark.parametrize("code", [c for _, c in DOC_SNIPPETS], ids=[i for i, _ in DOC_SNIPPETS])
def test_doc_snippet_runs_as_pasted(code):
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr
