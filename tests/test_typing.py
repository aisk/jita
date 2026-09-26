"""Static typing: the runtime classes behind the stubs, stub freshness, and
pyright, mypy and stubtest over the package, the examples and the
tests/typing corpus.

The accepted corpus (`tools/gen_stubs.py corpus`) has one call per
operand class tuple the encoder accepts and must produce no diagnostics.
The hand written `tests/typing/errors_*.py` files hold calls the stubs
must reject, each with the exact ignore comment of both checkers; unused
ignores are errors in both configurations, so a call that stops being
rejected fails the run.
"""

import ast
import ctypes
import importlib.util
import itertools
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

import jita.aarch64 as A
import jita.aarch64.insns
import jita.aarch64.regs
import jita.x64 as X
import jita.x64.insns
import jita.x64.regs
from jita import Assembler
from jita.aarch64.insns import Aarch64Assembler
from jita.x64.insns import X64Assembler
from jita.x64.mem import Mem8, Mem32, Mem64, MemAny, MemExpr

ROOT = Path(__file__).resolve().parent.parent
TYPING = ROOT / "tests" / "typing"
CHECKED = ["jita", "examples", "tests/typing"]


def _load_generator() -> ModuleType:
    spec = importlib.util.spec_from_file_location("gen_stubs", ROOT / "tools" / "gen_stubs.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def gen() -> ModuleType:
    return _load_generator()


# -- runtime classes -----------------------------------------------------------


def test_assembler_dispatches_to_the_arch_class():
    assert type(Assembler("x64")) is X64Assembler
    assert type(Assembler(X)) is X64Assembler
    assert type(Assembler(X.ARCH)) is X64Assembler
    assert type(Assembler("aarch64")) is Aarch64Assembler
    assert isinstance(Assembler("x64"), Assembler)
    assert type(Assembler()) is Assembler(None).arch.assembler_class
    assert X64Assembler().arch is X.ARCH and Aarch64Assembler().arch is A.ARCH
    with pytest.raises(TypeError, match="cannot assemble for aarch64"):
        X64Assembler("aarch64")


def test_x64_register_classes():
    r = jita.x64.regs
    assert all(type(x) is r.Gp64 for x in (r.rax, r.r15, r.gp64(3)))
    assert type(r.ah) is r.Gp8 and r.ah.high and not r.al.high
    assert (r.eax.kind, r.eax.size, r.eax.code) == ("gp", 4, 0)
    assert type(r.xmm3) is r.Xmm and type(r.ymm(2)) is r.Ymm and type(r.st(1)) is r.St
    assert type(r.rip) is r.Rip and r.rip.kind == "rip"


def test_x64_memory_classes():
    r = jita.x64.regs
    assert type(X.qword[r.rbx]) is Mem64 and type(X.byte[0x1000]) is Mem8
    assert type(X.ptr[r.rbx]) is MemAny and type(r.rbx + 8) is MemAny and type(8 * r.rcx) is MemAny
    assert type(X.qword[r.rbx] + 8) is Mem64 and type(X.qword[r.rbx] - 8) is Mem64
    assert type(X.dword[r.rbx + 8]) is Mem32
    # Equality compares the fields, not the class.
    assert X.qword[r.rbx] == MemExpr(base=r.rbx, size=8)
    assert hash(X.qword[r.rbx]) == hash(MemExpr(base=r.rbx, size=8))
    assert X.ptr[r.rbx] != X.qword[r.rbx]


def test_typed_scalars_are_sized_classes():
    class S(ctypes.Structure):
        _fields_ = [("a", ctypes.c_int32), ("b", ctypes.c_uint8), ("c", ctypes.c_double)]

    p = X.typed(jita.x64.regs.rdi, S)
    assert (type(p.a), type(p.b), type(p.c)) == (Mem32, Mem8, Mem64)
    assert type(p.addr) is MemAny
    assert type(X.typed(MemExpr(base=jita.x64.regs.rdi), S).addr) is MemAny


def test_aarch64_register_classes():
    r = jita.aarch64.regs
    assert type(r.x0) is r.X and type(r.xzr) is r.X and type(r.wzr) is r.W
    assert type(r.sp) is r.Sp and (r.sp.kind, r.sp.rt) == ("sp", "x")
    assert type(r.fp128(3)) is r.Q and type(r.d1) is r.D and type(r.s1) is r.S
    assert r.lr is r.x30 and r.fp is r.x29
    m = r.x1 << 3
    assert type(m) is r.RegMod and m.reg is r.x1


@pytest.mark.parametrize("regs", [jita.x64.regs, jita.aarch64.regs], ids=["x64", "aarch64"])
def test_static_register_assignments(regs):
    # Every register is a module attribute with its name, exported, and
    # the module __all__ is a literal the checkers can read.
    for name, reg in regs.ALL_REGS.items():
        assert getattr(regs, name) is reg, name
        assert reg.name == name
    assert set(regs.ALL_REGS) <= set(regs.__all__)
    source = Path(regs.__file__).read_text()
    assert _literal_all(source) == regs.__all__
    assert len(set(regs.__all__)) == len(regs.__all__)
    assert all(hasattr(regs, n) for n in regs.__all__)


# -- stubs ---------------------------------------------------------------------


def _literal_all(source: str) -> list[str]:
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets):
            value = ast.literal_eval(node.value)
            assert isinstance(value, list)
            return value
    raise AssertionError("no literal __all__")


def test_stubs_are_fresh(gen):
    for path, text in gen.generate().items():
        assert path.read_text() == text, f"{path.relative_to(ROOT)} is stale, run: uv run python tools/gen_stubs.py"


@pytest.mark.parametrize(
    "module", [X, jita.x64.insns, A, jita.aarch64.insns], ids=lambda m: m.__name__.removeprefix("jita.")
)
def test_stub_all_matches_runtime(module):
    stub = Path(module.__file__).with_suffix(".pyi")
    assert _literal_all(stub.read_text()) == list(module.__all__)


def test_generator_merges_into_a_partition(gen):
    points = [("A", "X"), ("B", "X"), ("A", "Y"), ("C", "Z")]
    sigs = gen.merge(points)
    covered = [p for s in sigs for p in itertools.product(*s)]
    assert sorted(covered) == sorted(points)


# -- checkers ------------------------------------------------------------------


def _available(module: str) -> str | None:
    """None if `python -m module --version` works, else why not."""
    if importlib.util.find_spec(module) is None:
        return f"{module} is not installed"
    try:
        proc = subprocess.run(
            [sys.executable, "-m", module, "--version"], cwd=ROOT, capture_output=True, text=True, timeout=300
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        return f"{module} cannot run: {e}"
    if proc.returncode != 0:
        return f"{module} cannot run: {(proc.stderr or proc.stdout).strip()[-500:]}"
    return None


@pytest.fixture(scope="module")
def checks(gen, tmp_path_factory) -> dict[str, tuple[int, str] | str]:
    """Run pyright, mypy and stubtest concurrently over the checked paths
    and the generated accepted corpus. Name -> (returncode, output), or
    the reason the checker cannot run."""
    out = tmp_path_factory.mktemp("typing")
    corpus = []
    for name, text in gen.corpus().items():
        path = out / name
        path.write_text(text)
        corpus.append(str(path))
    allowlist = str(TYPING / "stubtest_allowlist.txt")
    mypy_cache = str(ROOT / ".mypy_cache" / "mypy")
    commands = {
        "pyright": ("pyright", ["--outputjson", "-p", "pyproject.toml", *CHECKED, *corpus]),
        # stubtest runs at the same time and uses the default cache; sharing
        # one sqlite cache fails with "database is locked" on Windows.
        "mypy": ("mypy", ["--no-color-output", "--no-pretty", "--show-traceback", "--cache-dir", mypy_cache, *CHECKED, *corpus]),
        "stubtest": ("mypy.stubtest", ["--concise", "--allowlist", allowlist, "jita.x64", "jita.aarch64"]),
    }
    # The prefilter check runs alongside the checkers: about 5 s alone.
    generator = str(ROOT / "tools" / "gen_stubs.py")
    procs = {
        "prefilter": subprocess.Popen(
            [sys.executable, generator, "--unfiltered", "2"],
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    }
    results: dict[str, tuple[int, str] | str] = {}
    for name, (module, args) in commands.items():
        why = _available(module.split(".")[0])
        if why is not None:
            results[name] = why
            continue
        procs[name] = subprocess.Popen(
            [sys.executable, "-m", module, *args], cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
    for name, proc in procs.items():
        try:
            stdout, stderr = proc.communicate(timeout=600)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate()
            results[name] = f"{name} timed out"
            continue
        # pyright reports JSON on stdout; notices on stderr stay apart.
        results[name] = (proc.returncode, stdout if name == "pyright" else stdout + stderr)
    return results


def _result(checks: dict[str, tuple[int, str] | str], name: str) -> tuple[int, str]:
    res = checks[name]
    if isinstance(res, str):
        pytest.skip(res)
    return res


def test_pyright(checks):
    _, output = _result(checks, "pyright")
    try:
        report = json.loads(output)
    except json.JSONDecodeError:
        pytest.fail(f"pyright failed:\n{output}")
    diags = [
        f"{d['file']}:{d['range']['start']['line'] + 1}: {d['severity']}: {d['message']} ({d.get('rule', '')})"
        for d in report["generalDiagnostics"]
        if d["severity"] in ("error", "warning")
    ]
    assert not diags, "\n".join(diags)
    assert report["summary"]["filesAnalyzed"] > 40


def test_mypy(checks):
    code, output = _result(checks, "mypy")
    assert code == 0, output


def test_stubtest(checks):
    code, output = _result(checks, "stubtest")
    assert code == 0, output


def test_prefilter_matches_encoder(checks):
    # The generator's class level prefilter must not drop a class tuple the
    # encoder accepts (checked for up to 2 operands, without the filter).
    code, output = _result(checks, "prefilter")
    assert code == 0, output
