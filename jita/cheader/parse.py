"""pycparser with recovery: a top level declaration it cannot parse is
reported and taken out, and the text is parsed again."""

import re
from typing import Any

import pycparser  # type: ignore[import-untyped]

from ..core.errors import HeaderError
from .diagnostic import Diagnostic
from .prelude import UNKNOWN
from .rewrite import Rewritten

__all__ = ["parse"]

_ERROR = re.compile(r"<jita>:(\d+):(\d+): (.*)", re.S)
_WORD = re.compile(r"[A-Za-z_]\w*")
_NOT_NEWLINE = re.compile(r"[^\n]")
_POINTER_NAME = re.compile(r"[^(]*\(\s*\*+\s*([A-Za-z_]\w*)")
_DEFINED_TAG = re.compile(r"\b(struct|union)\s+([A-Za-z_]\w*)\s*\{")
_GROUP = re.compile(r"\([^()]*\)|\[[^\[\]]*\]")


def _typedef_names(chunk: str) -> list[str]:
    """The names a `typedef ...;` declares, one per declarator."""
    tail = chunk[chunk.rfind("}") + 1 :] if "}" in chunk else chunk[chunk.find("typedef") + 7 :]
    parts, depth, start = [], 0, 0
    tail = tail.rstrip().rstrip(";")
    for i, c in enumerate(tail):
        if c in "([":
            depth += 1
        elif c in ")]":
            depth -= 1
        elif c == "," and depth == 0:
            parts.append(tail[start:i])
            start = i + 1
    parts.append(tail[start:])
    names = []
    for part in parts:
        if m := _POINTER_NAME.match(part):
            names.append(m.group(1))
            continue
        while (flat := _GROUP.sub(" ", part)) != part:
            part = flat
        if words := _WORD.findall(part):
            names.append(words[-1])
    return names


def parse(
    rw: Rewritten, table: list[tuple[str, int] | None], strict: bool, diags: list[Diagnostic]
) -> tuple[Any, set[str], set[str]]:
    """Parse `rw.text`; returns the AST, the typedef names whose
    declarations were replaced after a parse error and the struct and
    union classes (`struct_X`) whose definitions were taken out."""
    parser: Any = pycparser.CParser()
    done: set[int] = set()
    recovered: set[str] = set()
    lost: set[str] = set()
    while True:
        try:
            return parser.parse(rw.text, "<jita>"), recovered, lost
        except pycparser.c_parser.ParseError as e:
            m = _ERROR.match(str(e))
            if m:
                line, column, what = int(m.group(1)), int(m.group(2)), m.group(3).strip()
            else:
                # A few errors carry no coordinate: use the next token's.
                try:
                    coord = parser._tok_coord(parser._peek())
                    line, column, what = coord.line, coord.column, str(e).split(": ", 1)[-1]
                except Exception:
                    raise HeaderError(f"cannot parse the preprocessed header: {e}") from None
            source = table[line] if line < len(table) else None
            file, src_line = source if source else (None, None)
            where = f"{file}:{src_line}: " if file else ""
            index = rw.span_index(rw.offset(line, column))
            chunk = rw.text[slice(*rw.spans[index])] if index is not None else ""
            snippet = " ".join(chunk.split())[:80]
            if strict:
                raise HeaderError(f"{where}{what}: {snippet}") from None
            if index is None or index in done:
                raise HeaderError(f"{where}cannot recover from a parse error: {what}") from None
            done.add(index)
            lost.update(f"{kind}_{tag}" for kind, tag in _DEFINED_TAG.findall(chunk))
            names: list[str] = []
            replacement = _NOT_NEWLINE.sub(" ", chunk)
            if chunk.split(None, 1)[:1] == ["typedef"]:
                # Keep the names as types of unknown size, so that the rest
                # still parses and uses by pointer still load.
                names = [n for n in _typedef_names(chunk) if n != UNKNOWN]
                if names:
                    body = chunk.lstrip()
                    lead = chunk[: len(chunk) - len(body)]
                    typedefs = " ".join(f"typedef {UNKNOWN} {n};" for n in names)
                    replacement = lead + typedefs + "\n" * body.count("\n")
                    recovered.update(names)
            name = names[0] if len(names) == 1 else None
            diags.append(Diagnostic("skip", "parse-error", name, file, src_line, f"{what}: {snippet}"))
            rw.replace_span(index, replacement)
