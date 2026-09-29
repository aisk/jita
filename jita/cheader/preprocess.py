"""The C preprocessor: pcpp with the host's predefined macros, C semantics
for unknown identifiers, integer suffixes in `#if`, `__has_include`, and
system headers that are skipped instead of read."""

import copy
import os
import re
from collections.abc import Mapping, Sequence
from typing import Any, NoReturn

import pcpp  # type: ignore[import-untyped]

from ..core.errors import HeaderError
from .diagnostic import Diagnostic, DiagnosticKind
from .prelude import PRELUDE, predefined_macros

__all__ = ["Preprocessed", "preprocess"]

# The pseudo file the translation unit is parsed as; the prelude and the
# user's defines live in it, so its names are never exposed.
TU_NAME = "<jita>"

_PP_NUMBER = re.compile(r"\.?\d(?:[eEpP][+-]|[\w.])*")
_INTEGER = re.compile(r"(?:0[xX][0-9a-fA-F]+|0[bB][01]+|\d+)[uUlL]*")


# pcpp has no type information; the hooks are overridden freely.
_Base: Any = pcpp.Preprocessor


class Preprocessor(_Base):
    def __init__(self, include_dirs: Sequence[str], diags: list[Diagnostic]):
        super().__init__()
        self.lexer: Any = _NumberLexer(self.lexer)
        self.path = list(include_dirs)
        # pcpp rewrites paths under the cwd to relative ones; keep them absolute.
        self.rewrite_paths: list[tuple[str, str]] = []
        self.diags = diags

    def note(self, kind: DiagnosticKind, file: object, line: object, message: str) -> None:
        path = os.path.abspath(file) if isinstance(file, str) and file else None
        self.diags.append(Diagnostic("note", kind, None, path, line if isinstance(line, int) else None, message))

    def on_error(self, file: str, line: int, msg: str) -> None:
        self.note("preprocessor", file, line, msg)

    def on_include_not_found(
        self, is_malformed: bool, is_system_include: bool, curdir: str, includepath: str
    ) -> NoReturn:
        d = self.lastdirective
        if is_system_include and not is_malformed:
            self.note("missing-include", d.source, d.lineno, f"<{includepath}> is not in include_dirs, ignored")
            raise pcpp.OutputDirective(pcpp.Action.IgnoreAndRemove)
        where = f"{os.path.abspath(d.source)}:{d.lineno}: " if d is not None and d.source else ""
        raise HeaderError(f'{where}cannot find "{includepath}" in the including directory or include_dirs')

    def on_unknown_macro_in_expr(self, ident: str) -> int:
        return 0

    def on_unknown_macro_in_defined_expr(self, tok: Any) -> bool:
        return tok.value in ("__has_include", "__has_include_next")

    def on_directive_unknown(self, directive: Any, toks: list[Any], ifpassthru: bool, precedingtoks: list[Any]) -> Any:
        name = directive.value
        if name in ("error", "warning"):
            text = "".join(t.value for t in toks).strip()
            self.note("preprocessor", directive.source, directive.lineno, f"#{name} {text}")
            return True
        if name in ("include_next", "import", "ident", "sccs", "assert", "unassert"):
            self.note("preprocessor", directive.source, directive.lineno, f"#{name} is not supported, ignored")
            return True
        return super().on_directive_unknown(directive, toks, ifpassthru, precedingtoks)

    def has_include(self, name: str) -> bool:
        return any(os.path.isfile(os.path.join(d, name)) for d in [*self.temp_path[:1], *self.path])

    def evalexpr(self, tokens: list[Any]) -> Any:
        # pcpp's evaluator compares ULL values as signed and does not know
        # LL: keep only a U suffix. __has_include(<x>) and
        # __has_include("x") look in include_dirs.
        out = []
        i = 0
        while i < len(tokens):
            tok = tokens[i]
            if tok.type == self.t_INTEGER:
                text = tok.value
                body = text.rstrip("uUlL")
                unsigned = "u" in text[len(body) :].lower()
                if body[:2] in ("0b", "0B"):
                    body = str(int(body, 2))
                tok = copy.copy(tok)
                tok.value = self.t_INTEGER_TYPE(body + ("U" if unsigned else ""))
            elif tok.type == self.t_ID and tok.value in ("__has_include", "__has_include_next"):
                j = i + 1
                while j < len(tokens) and tokens[j].type in self.t_WS:
                    j += 1
                if j == len(tokens) or tokens[j].value != "(":
                    # defined(__has_include) is answered by the hook above.
                    out.append(tok)
                    i += 1
                    continue
                k = j
                while k < len(tokens) and tokens[k].value != ")":
                    k += 1
                name = "".join(t.value for t in tokens[j + 1 : k]).strip().strip('<>"')
                tok = copy.copy(tok)
                tok.type = self.t_INTEGER
                tok.value = self.t_INTEGER_TYPE("1" if self.has_include(name) else "0")
                i = k
            out.append(tok)
            i += 1
        return super().evalexpr(out)


class _NumberLexer:
    """pcpp's lexer, except that a C preprocessing number (`8LL`, `2.5f`,
    `0x1p4`, `1e+3`) is one token. pcpp alone splits `8LL` into `8L` and
    the identifier `L`, which a macro named `L` then expands."""

    def __init__(self, inner: Any):
        object.__setattr__(self, "_inner", inner)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(self._inner, name, value)

    def clone(self) -> "_NumberLexer":
        return _NumberLexer(self._inner.clone())

    def token(self) -> Any:
        inner = self._inner
        tok = inner.token()
        if tok is None:
            return None
        data = inner.lexdata
        at = tok.lexpos
        if tok.type in ("CPP_INTEGER", "CPP_FLOAT") or (tok.type == "CPP_DOT" and data[at + 1 : at + 2].isdigit()):
            m = _PP_NUMBER.match(data, at)
            if m:
                tok.value = m.group()
                tok.type = "CPP_INTEGER" if _INTEGER.fullmatch(tok.value) else "CPP_FLOAT"
                inner.lexpos = m.end()
        return tok


class Preprocessed:
    """The preprocessed text with `#line` directives, the path of the
    translation unit's pseudo file and the object like macros defined in
    the user's headers as (name, expansion, file, line)."""

    def __init__(self, text: str, tu_path: str, macros: list[tuple[str, str, str, int]]):
        self.text = text
        self.tu_path = tu_path
        self.macros = macros


def _write(p: Preprocessor) -> str:
    """The output text with a `#line` directive wherever the source line
    does not simply follow the previous one. pcpp's own writer emits none
    when a header is included twice in a row, and its line numbers then
    point into the first copy."""
    out: list[str] = []
    source, line = None, 0
    toks: list[Any] = []
    while True:
        tok = p.token()
        if tok is not None:
            if tok.type not in p.t_LINECONT:
                toks.append(tok)
            if not tok.value.startswith("\n"):
                continue
        if any(t.type not in p.t_WS for t in toks):
            first = toks[0]
            gap = first.lineno - line - 1
            if first.source != source or not 0 <= gap <= 8:
                out.append(f'#line {first.lineno} "{first.source}"\n')
            else:
                out.append("\n" * gap)
            # A `_Pragma("x")` becomes a `#pragma x` line of its own, as in
            # GCC, and the rest of the line continues after a `#line`.
            for i, part in enumerate(_split_pragmas(p, toks)):
                if not i % 2:
                    out.append("".join(t.value for t in part))
                elif part is not None:
                    out.append(f'\n#pragma {part}\n#line {first.lineno} "{first.source}"\n')
            if not out[-1].endswith("\n"):
                out.append("\n")
            source, line = first.source, first.lineno
        toks = []
        if tok is None:
            return "".join(out)


def _pragma_operator(p: Preprocessor, toks: list[Any], i: int) -> tuple[str, int] | None:
    """The pragma text of `_Pragma("text")` or MSVC's `__pragma(text)` at
    `toks[i]`, and the index after it; None if there is none."""
    if toks[i].type != p.t_ID or toks[i].value not in ("_Pragma", "__pragma"):
        return None
    j = i + 1
    while j < len(toks) and toks[j].type in p.t_WS:
        j += 1
    if j == len(toks) or toks[j].value != "(":
        return None
    depth = 0
    for k in range(j, len(toks)):
        depth += {"(": 1, ")": -1}.get(toks[k].value, 0)
        if depth == 0:
            inner = [t for t in toks[j + 1 : k] if t.type not in p.t_COMMENT]
            if toks[i].value == "__pragma":
                return "".join(t.value for t in inner).strip(), k + 1
            strings = [t for t in inner if t.type not in p.t_WS]
            if len(strings) != 1 or strings[0].type != p.t_STRING:
                return None
            literal = strings[0].value
            return literal[literal.index('"') + 1 : -1].replace('\\"', '"').replace("\\\\", "\\"), k + 1
    return None


# Pragmas that cannot change a layout; the operator forms of these are
# dropped instead of becoming `#pragma` lines, since they also appear
# where a line break is not valid C (MSVC's Py_UNUSED puts
# `__pragma(warning(...))` in parameter lists).
_HARMLESS_PRAGMA = re.compile(r"(?:warning|message|comment|once|region|endregion|(?:GCC|clang) (?:diagnostic|visibility|poison|system_header))\b")


def _split_pragmas(p: Preprocessor, toks: list[Any]) -> list[Any]:
    """`toks` split at each `_Pragma("text")` and `__pragma(text)`: token
    lists at even positions, the pragma texts between them. Harmless
    pragmas are dropped."""
    parts: list[Any] = []
    start = i = 0
    while i < len(toks):
        found = _pragma_operator(p, toks, i)
        if found is None:
            i += 1
            continue
        text, end = found
        if _HARMLESS_PRAGMA.match(text):
            parts.append(toks[start:i])
            parts.append(None)
        else:
            parts += [toks[start:i], text]
        start = i = end
    parts.append(toks[start:])
    return parts


def _define_line(name: str, value: int | str | None) -> str:
    if not name.isidentifier():
        raise ValueError(f"defines: {name!r} is not a macro name")
    if value is None:
        return f"#undef {name}"
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise TypeError(f"defines: the value of {name} must be an int, a str or None, got {value!r}")
    return f"#define {name} {value}"


def preprocess(
    headers: Sequence[str],
    include_dirs: Sequence[str],
    defines: Mapping[str, int | str | None],
    diags: list[Diagnostic],
) -> Preprocessed:
    """Preprocess the absolute paths `headers` as one translation unit."""
    p = Preprocessor(include_dirs, diags)
    for d in predefined_macros():
        p.define(d)
    tu = "\n".join(
        [
            PRELUDE,
            *(_define_line(k, v) for k, v in defines.items()),
            *(f'#include "{h.replace(os.sep, "/")}"' for h in headers),
            "",
        ]
    )
    p.parse(tu, TU_NAME)
    text = _write(p)
    tu_path = os.path.abspath(TU_NAME)
    macros = []
    for name, m in p.macros.items():
        source = m.source
        if m.arglist is not None or not source or os.path.abspath(source) == tu_path:
            continue
        # pcpp expands in place, so expand a copy.
        toks = p.expand_macros(copy.deepcopy(list(m.value)))
        macros.append((name, "".join(t.value for t in toks).strip(), os.path.abspath(source), m.lineno))
    return Preprocessed(text, tu_path, macros)
