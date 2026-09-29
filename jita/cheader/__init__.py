"""Load C headers at runtime as ctypes types.

    h = load("Python.h", include_dirs=[sysconfig.get_path("include")])
    typed(rdi, h.PyVarObject).ob_size

`load` preprocesses the headers with pcpp, parses them with pycparser
and builds a module like `Header` with a `ctypes.Structure` or `Union`
per struct and union, the typedefs, enum classes and members, function
prototypes as CFUNCTYPE classes and the object like macros with a
constant value. Layouts follow the host ABI, as computed by ctypes.
Declarations that cannot be converted exactly are skipped and reported
through `diagnostics`, never loaded with a wrong layout.

Needs the optional dependencies: `pip install "jita[cheader]"`.
"""

import fnmatch
import importlib
import os
import sys
import sysconfig
import types as _types
from collections.abc import Iterable, Mapping, Sequence

from ..core.errors import HeaderError
from ..core.structs import is_dunder
from .diagnostic import Diagnostic, DiagnosticKind

__all__ = ["Diagnostic", "DiagnosticKind", "Header", "HeaderError", "diagnostics", "load", "python_defines"]


class Header(_types.ModuleType):
    """The names of loaded C headers, as module attributes.

    `struct foo` is `struct_foo`, `union foo` is `union_foo`, `enum foo`
    is `enum_foo`; typedefs, enum members, functions and macros keep
    their names. A name that was skipped raises an AttributeError that
    says why. It has no other attributes, so C names never clash with
    them; `diagnostics(h)` gives the report.
    """


def diagnostics(h: Header) -> list[Diagnostic]:
    """What `load` skipped, ignored or degraded while building `h`."""
    return list(getattr(h, "__diagnostics__"))


def python_defines() -> dict[str, int | str | None]:
    """The build macros of the running interpreter that change struct
    layouts, for `load("Python.h", ..., defines=python_defines())`.

    pyconfig.h does not always record them: on Windows one pyconfig.h
    serves the GIL and free threaded builds, and Py_GIL_DISABLED must come
    from the compiler command line. The headers test them with #ifdef, so
    only the ones this build has are returned, each as 1.
    """
    flags = {
        "Py_GIL_DISABLED": False,
        # sysconfig on Windows may lack these; the sys functions exist only
        # in such builds.
        "Py_DEBUG": hasattr(sys, "gettotalrefcount"),
        "Py_TRACE_REFS": hasattr(sys, "getobjects"),
        "Py_STATS": False,
    }
    return {name: 1 for name, fallback in flags.items() if sysconfig.get_config_var(name) or fallback}


def _find(header: str, include_dirs: Sequence[str]) -> str:
    for d in ["", *include_dirs]:
        path = os.path.join(d, header)
        if os.path.isfile(path):
            return os.path.abspath(path)
    where = f" or in {', '.join(include_dirs)}" if include_dirs else ""
    raise HeaderError(f"cannot find header {header!r}{where}")


def load(
    *headers: str | os.PathLike[str],
    include_dirs: Sequence[str | os.PathLike[str]] = (),
    defines: Mapping[str, int | str | None] | None = None,
    types: Mapping[str, type] | None = None,
    names: Iterable[str] | None = None,
    exclude: Iterable[str] = (),
    strict: bool = False,
) -> Header:
    """Load C headers as one translation unit into a new `Header`.

    Each header is looked up as given, then in `include_dirs`, which also
    serve the headers' own `#include` and `__has_include`. `<...>`
    includes that are not found are skipped: system headers are replaced
    by a built in prelude. `defines` adds macros (None undefines one),
    `types` maps type names to ctypes types ahead of the headers'
    typedefs, `names` and `exclude` are fnmatch patterns choosing the
    names to expose. With `strict`, any skipped declaration raises
    `HeaderError` instead of being recorded.
    """
    for dependency in ("pycparser", "pcpp"):
        try:
            importlib.import_module(dependency)
        except ImportError:
            raise ImportError("jita.cheader needs pycparser and pcpp, install jita[cheader]") from None
    from .convert import Converter
    from .parse import parse
    from .preprocess import preprocess
    from .rewrite import rewrite, strip_line_directives

    if not headers:
        raise TypeError("load() needs at least one header")
    dirs = [os.path.abspath(os.fspath(d)) for d in include_dirs]
    paths = [_find(os.fspath(h), dirs) for h in headers]
    stem = os.path.splitext(os.path.basename(paths[0]))[0]
    module = f"jita.cheader.{stem}"

    diags: list[Diagnostic] = []
    pre = preprocess(paths, dirs, dict(defines or {}), diags)
    text, table = strip_line_directives(pre.text, pre.tu_path)
    rw = rewrite(text)
    ast, recovered, lost = parse(rw, table, strict, diags)
    conv = Converter(rw, table, pre.tu_path, module, dict(types or {}), recovered, lost, strict, diags)
    conv.run(ast)
    conv.evaluate_macros(pre.macros)

    include = list(names) if names is not None else None
    excluded = list(exclude)
    h = Header(module)
    h.__file__ = paths[0]
    skipped: dict[str, Diagnostic] = {}
    for d in diags:
        if d.severity == "skip" and d.name is not None:
            skipped.setdefault(d.name, d)
    namespace = vars(h)
    count = 0
    for name, value in conv.ns.items():
        if name in conv.prelude:
            continue
        if isinstance(value, type) and value in conv.failed:
            if value.__name__ in skipped:
                skipped.setdefault(name, skipped[value.__name__])
            continue
        if is_dunder(name):
            diags.append(
                Diagnostic("note", "shadowed-member", name, None, None, "a dunder name is reserved by the module")
            )
            continue
        if include is not None and not any(fnmatch.fnmatchcase(name, p) for p in include):
            continue
        if any(fnmatch.fnmatchcase(name, p) for p in excluded):
            continue
        namespace[name] = value
        count += 1

    def __getattr__(name: str) -> object:
        d = skipped.get(name)
        if d is not None:
            where = f" ({os.path.basename(d.file)}:{d.line})" if d.file else ""
            raise AttributeError(f"{name} was skipped: {d.message}{where}")
        raise AttributeError(f"module {module!r} has no attribute {name!r}")

    namespace["__getattr__"] = __getattr__
    namespace["__diagnostics__"] = diags
    h.__doc__ = f"{count} names from {', '.join(paths)}, {sum(d.severity == 'skip' for d in diags)} skipped"
    return h
