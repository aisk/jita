"""Text passes between the preprocessor and the parser.

`strip_line_directives` blanks the `#line` directives and keeps a table
from output line to source file and line. `rewrite` is a single scan over
the result that removes GNU and MSVC attributes, turns the struct level
`packed` and `aligned(N)` into marker fields, records the other
attributes in a side table, replaces function bodies with `;` and
records the top level declaration spans and the brace structure. Line
numbers never change, so parser coordinates map straight back through
the `#line` table.
"""

import bisect
import re
from dataclasses import dataclass, field

from .prelude import HARMLESS_ATTRIBUTES, LAYOUT_ATTRIBUTES

__all__ = ["MARKER", "Attr", "Rewritten", "rewrite", "strip_line_directives"]

# Marker fields are `char __jita_attr_packed;` and `char __jita_attr_aligned_16;`.
MARKER = "__jita_attr_"

_LINE = re.compile(r'[ \t]*#(?:line)?[ \t]+(\d+)[ \t]+"(.*)"')

_TOKEN = re.compile(
    r"""
      (?P<pp>^[ \t]*\#[^\n]*)
    | (?P<str>"(?:\\.|[^"\\\n])*"|'(?:\\.|[^'\\\n])*')
    | (?P<word>\w+)
    | (?P<punct>[^\s\w])
    """,
    re.M | re.X,
)
_BRACE = re.compile(r"""^[ \t]*\#[^\n]*|"(?:\\.|[^"\\\n])*"|'(?:\\.|[^'\\\n])*'|[{}]""", re.M)
_PAREN = re.compile(r"""\s*\(""")
_PARENS = re.compile(r"""[()]|"(?:\\.|[^"\\\n])*"|'(?:\\.|[^'\\\n])*'""")
_ALIGNED = re.compile(r"(?:aligned|align)\s*\(\s*(0[xX][0-9a-fA-F]+|[1-9][0-9]*)[uUlL]*\s*\)")
_NOT_NEWLINE = re.compile(r"[^\n]")

_ATTRIBUTE_WORDS = frozenset(["__attribute__", "__attribute", "__declspec"])
_AGGREGATE_WORDS = frozenset(["struct", "union"])


def strip_line_directives(text: str, default: str) -> tuple[str, list[tuple[str, int] | None]]:
    """Replace every `#line` directive with an empty line. `table[n]` is the
    (file, line) of output line n (1 based), None for a directive line."""
    table: list[tuple[str, int] | None] = [None]
    lines = text.split("\n")
    file, line = default, 1
    for i, ln in enumerate(lines):
        m = _LINE.match(ln) if ln.startswith(("#", " ", "\t")) else None
        if m:
            file, line = m.group(2), int(m.group(1))
            lines[i] = ""
            table.append(None)
            continue
        table.append((file, line))
        line += 1
    return "\n".join(lines), table


@dataclass
class Attr:
    """An attribute the rewrite did not drop, at output offset `offset`.

    `kind` is "packed" or "aligned" for a field level layout attribute
    inside the struct or union body whose `{` is at `body`, or
    "unsupported" for anything that makes its declaration unloadable.
    `start` and `end` delimit the member declaration it is part of.
    """

    kind: str
    text: str
    offset: int
    value: int = 0
    body: int = -1
    start: int = 0
    end: int = 0


class Rewritten:
    """The rewritten text and the tables the parser and converter use."""

    def __init__(self, text: str, attrs: list[Attr], spans: list[tuple[int, int]], braces: list[tuple[int, int]]):
        self.text = text
        self.attrs = attrs
        self.spans = spans
        self.span_starts = [s for s, _ in spans]
        # braces: (offset of a brace, offset of the innermost open `{` after it)
        self.brace_offsets = [o for o, _ in braces]
        self.brace_inner = [inner for _, inner in braces]
        self.line_starts = [0, *(m.end() for m in re.finditer("\n", text))]

    def replace_span(self, index: int, replacement: str) -> None:
        """Replace the text of top level span `index`, which must keep its
        line count, and move every recorded offset after it."""
        start, end = self.spans[index]
        delta = len(replacement) - (end - start)
        self.text = self.text[:start] + replacement + self.text[end:]
        self.spans[index] = (start, start + len(replacement))
        if not delta:
            return

        def move(offset: int) -> int:
            return offset + delta if offset >= end else offset

        self.spans[index + 1 :] = [(move(a), move(b)) for a, b in self.spans[index + 1 :]]
        self.span_starts = [a for a, _ in self.spans]
        self.brace_offsets = [move(o) for o in self.brace_offsets]
        self.brace_inner = [move(o) for o in self.brace_inner]
        for at in self.attrs:
            at.offset, at.body, at.start, at.end = move(at.offset), move(at.body), move(at.start), move(at.end)
        self.line_starts = [move(o) for o in self.line_starts]

    def offset(self, line: int, column: int) -> int:
        """The offset of a 1 based parser coordinate."""
        return self.line_starts[line - 1] + column - 1

    def span_index(self, offset: int) -> int | None:
        i = bisect.bisect_right(self.span_starts, offset) - 1
        return i if i >= 0 and offset < self.spans[i][1] else None

    def body_of(self, offset: int) -> int:
        """The offset of the innermost `{` open at `offset`, or -1."""
        i = bisect.bisect_right(self.brace_offsets, offset) - 1
        return self.brace_inner[i] if i >= 0 else -1


def _attribute_items(text: str) -> list[str]:
    """Split the attribute list `packed, aligned(8)` at top level commas."""
    items, depth, start = [], 0, 0
    for i, c in enumerate(text):
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
        elif c == "," and depth == 0:
            items.append(text[start:i].strip())
            start = i + 1
    items.append(text[start:].strip())
    return [it for it in items if it]


def _attribute_name(item: str) -> str:
    m = re.match(r"\w+", item)
    name = m.group(0) if m else item
    if len(name) > 4 and name.startswith("__") and name.endswith("__"):
        name = name[2:-2]
    return name


def _close_paren(text: str, i: int) -> int:
    """The offset after the `)` matching the `(` at `i`, or -1."""
    depth = 0
    for m in _PARENS.finditer(text, i):
        c = m.group(0)
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                return m.end()
    return -1


@dataclass
class _Open:
    """An open brace: what it opens, its offset, where the current member
    declaration in it started and the field attributes waiting for the
    `;` that ends it."""

    kind: str
    offset: int
    member: int
    waiting: list[Attr] = field(default_factory=list)

    def end_member(self, pos: int) -> None:
        for at in self.waiting:
            at.end = pos
        self.waiting = []


class _Scanner:
    def __init__(self, text: str):
        self.text = text
        self.edits: list[tuple[int, int, str]] = []
        self.attrs: list[Attr] = []
        self.spans: list[tuple[int, int]] = []
        self.braces: list[tuple[int, bool]] = []
        self.stack: list[_Open] = []
        self.parens = 0
        self.prev = ";"
        self.prev2 = ";"
        self.last_close_kind = ""
        self.last_close = -1
        self.span_start = 0
        # An `=` in the current top level declaration: `(T){...}` after it
        # is a compound literal, not a function body.
        self.assigns = False
        # Struct level markers waiting for their `{`, and what the next
        # significant token must be: "struct" (a struct or union keyword),
        # "{" or "" (anything up to the aggregate's `{`).
        self.pending: list[str] = []
        self.pending_at = 0
        self.pending_text = ""
        self.expect = ""

    def unsupported(self, text: str, offset: int) -> None:
        self.attrs.append(Attr("unsupported", text, offset))

    def fail_pending(self) -> None:
        self.unsupported(self.pending_text, self.pending_at)
        self.pending = []
        self.expect = ""

    def attribute(self, word: str, start: int, after_word: int) -> int:
        """Handle the attribute starting at `start`; returns where to go on."""
        text = self.text
        m = _PAREN.match(text, after_word)
        if not m:
            return after_word
        end = _close_paren(text, m.end() - 1)
        if end < 0:
            return after_word
        inner = text[m.end() : end - 1].strip()
        if word != "__declspec" and inner.startswith("(") and inner.endswith(")"):
            inner = inner[1:-1]
        self.edits.append((start, end, _NOT_NEWLINE.sub(" ", text[start:end])))
        markers = []
        for item in _attribute_items(inner):
            name = _attribute_name(item)
            if name in HARMLESS_ATTRIBUTES:
                continue
            if name == "packed" and item.strip("_ ") == "packed":
                markers.append((f"{MARKER}packed", "packed", 0, item))
                continue
            if name in LAYOUT_ATTRIBUTES and (am := _ALIGNED.fullmatch(item.replace("__aligned__", "aligned"))):
                n = int(am.group(1), 0)
                markers.append((f"{MARKER}aligned_{n}", "aligned", n, item))
                continue
            self.unsupported(item, start)
        if markers:
            self.place(markers, start)
        return end

    def place(self, markers: list[tuple[str, str, int, str]], start: int) -> None:
        fields = [f"char {name};" for name, _, _, _ in markers]
        described = ", ".join(item for _, _, _, item in markers)
        prev, prev2 = self.prev, self.prev2
        if prev == "}" and self.last_close_kind == "aggregate":
            self.edits.append((self.last_close, self.last_close, " ".join(["", *fields, ""])))
            return
        if prev in _AGGREGATE_WORDS or (_is_word(prev) and prev2 in _AGGREGATE_WORDS):
            self.pending += fields
            self.pending_at, self.pending_text = start, described
            self.expect = "" if prev in _AGGREGATE_WORDS else "{"
            return
        if prev != "}" and not self.stack and self.parens == 0:
            self.pending += fields
            self.pending_at, self.pending_text = start, described
            self.expect = "struct"
            return
        if self.stack and self.stack[-1].kind == "aggregate" and prev != "}":
            for _, kind, n, item in markers:
                top = self.stack[-1]
                at = Attr(kind, item, start, n, top.offset, top.member)
                self.attrs.append(at)
                top.waiting.append(at)
            return
        self.unsupported(described, start)

    def open_brace(self, pos: int) -> None:
        prev, prev2 = self.prev, self.prev2
        if prev in _AGGREGATE_WORDS or (_is_word(prev) and prev2 in _AGGREGATE_WORDS):
            kind = "aggregate"
        elif prev == "enum" or (_is_word(prev) and prev2 == "enum"):
            kind = "enum"
        else:
            kind = "other"
        if self.pending:
            if kind == "aggregate":
                self.edits.append((pos + 1, pos + 1, " ".join(["", *self.pending, ""])))
                self.pending = []
                self.expect = ""
            else:
                self.fail_pending()
        self.stack.append(_Open(kind, pos, pos + 1))
        self.braces.append((pos, True))

    def close_brace(self, pos: int) -> None:
        if self.stack:
            top = self.stack.pop()
            top.end_member(pos)
            self.last_close_kind = top.kind
        self.last_close = pos
        self.braces.append((pos, False))

    def function_body(self, pos: int) -> int:
        """Replace the body `{...}` at `pos` with `;`; returns its end."""
        depth = 0
        for m in _BRACE.finditer(self.text, pos):
            c = m.group(0)
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    end = m.end()
                    self.edits.append((pos, end, ";" + "\n" * self.text.count("\n", pos, end)))
                    self.spans.append((self.span_start, end))
                    self.span_start = end
                    return end
        return len(self.text)

    def run(self) -> None:
        text = self.text
        pos = 0
        search = _TOKEN.search
        while (m := search(text, pos)) is not None:
            pos = m.end()
            group = m.lastgroup
            tok = m.group()
            start = m.start()
            if group == "pp":
                if not self.stack and not self.parens and not text[self.span_start : start].strip():
                    self.span_start = pos
                continue
            if group == "word" and tok in _ATTRIBUTE_WORDS:
                pos = self.attribute(tok, start, pos)
                continue
            if self.expect:
                if self.expect == "struct" and tok in _AGGREGATE_WORDS:
                    self.expect = ""
                elif self.expect == "{" and tok == "{":
                    self.expect = ""
                else:
                    self.fail_pending()
            if tok == "{":
                if not self.stack and not self.parens and self.prev == ")" and not self.assigns:
                    pos = self.function_body(start)
                    self.prev2, self.prev = self.prev, ";"
                    continue
                self.open_brace(start)
            elif tok == "}":
                self.close_brace(start)
            elif tok == "=" and not self.stack and not self.parens:
                self.assigns = True
            elif tok == "(":
                self.parens += 1
            elif tok == ")":
                self.parens -= 1
            elif tok == ";":
                if self.pending:
                    self.fail_pending()
                if not self.stack and not self.parens:
                    self.spans.append((self.span_start, pos))
                    self.span_start = pos
                    self.assigns = False
                elif self.stack:
                    self.stack[-1].end_member(start)
                    self.stack[-1].member = pos
            self.prev2, self.prev = self.prev, tok
        if self.pending:
            self.fail_pending()
        if text[self.span_start :].strip():
            self.spans.append((self.span_start, len(text)))


def _is_word(tok: str) -> bool:
    return tok[:1].isalpha() or tok[:1] == "_"


def rewrite(text: str) -> Rewritten:
    s = _Scanner(text)
    s.run()
    edits = sorted(s.edits, key=lambda e: e[0])
    pieces, last = [], 0
    ends, shifts, shift = [], [], 0
    for start, end, repl in edits:
        pieces += [text[last:start], repl]
        last = end
        shift += len(repl) - (end - start)
        ends.append(end)
        shifts.append(shift)

    def move(offset: int) -> int:
        i = bisect.bisect_right(ends, offset)
        return offset + (shifts[i - 1] if i else 0)

    pieces.append(text[last:])
    for at in s.attrs:
        at.offset, at.body, at.start, at.end = move(at.offset), move(at.body), move(at.start), move(at.end)
    spans = [(move(a), move(b)) for a, b in s.spans]
    braces: list[tuple[int, int]] = []
    open_stack: list[int] = []
    for offset, is_open in s.braces:
        offset = move(offset)
        if is_open:
            open_stack.append(offset)
        elif open_stack:
            open_stack.pop()
        braces.append((offset, open_stack[-1] if open_stack else -1))
    return Rewritten("".join(pieces), s.attrs, spans, braces)
