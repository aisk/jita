"""The pycparser AST to ctypes: one pass in source order, C semantics
constant evaluation and macro values.

C guarantees a type is complete before it is used by value, so every
struct is finished at its definition. Classes are created lazily, on the
first mention of their tag, because pointers to them may come first;
`completed` is the only test for "usable by value", since ctypes silently
gives size 0 to a structure without `_fields_`.
"""

import ctypes
import re
from typing import Any

import pycparser  # type: ignore[import-untyped]
from pycparser import c_ast

from ..core.errors import HeaderError
from .diagnostic import Diagnostic, DiagnosticKind, Severity
from . import prelude
from .prelude import STD_TYPES, UNKNOWN
from .rewrite import MARKER, Attr, Rewritten

__all__ = ["Converter"]

_AGGREGATE = (ctypes.Structure, ctypes.Union)
_INT_BITS = 8 * ctypes.sizeof(ctypes.c_int)
_LONG_BITS = 8 * ctypes.sizeof(ctypes.c_long)
_INT_CODES = "bBhHiIlLqQ"
# Enum members that would shadow these on the class are top level only.
_C_INT_ATTRS = frozenset(dir(ctypes.c_int)) | frozenset(dir(type(ctypes.c_int)))


def _base_type(names: list[str]) -> type | None:
    """The ctypes type of a list of C type specifier words; None for void."""
    words = set(names)
    longs = names.count("long")
    unsigned = "unsigned" in words
    if not words <= {
        "void",
        "_Bool",
        "char",
        "short",
        "int",
        "long",
        "float",
        "double",
        "signed",
        "unsigned",
        "_Complex",
    }:
        raise Unsupported(f"type {' '.join(names)} is not supported")
    complex_ = "_Complex" in words
    if words == {"void"}:
        return None
    if words == {"_Bool"}:
        return ctypes.c_bool
    t: type | None
    if "float" in words or "double" in words or (complex_ and words <= {"_Complex", "long"}):
        if "float" in words:
            t = getattr(ctypes, "c_float_complex", None) if complex_ else ctypes.c_float
        elif longs:
            t = getattr(ctypes, "c_longdouble_complex", None) if complex_ else ctypes.c_longdouble
        else:
            t = getattr(ctypes, "c_double_complex", None) if complex_ else ctypes.c_double
        if t is None:
            raise Unsupported(f"{' '.join(names)} has no ctypes type on this Python")
        return t
    if complex_:
        raise Unsupported(f"type {' '.join(names)} is not supported")
    if "char" in words:
        return ctypes.c_ubyte if unsigned else ctypes.c_byte if "signed" in words else ctypes.c_char
    if "short" in words:
        return ctypes.c_ushort if unsigned else ctypes.c_short
    if longs == 1:
        return ctypes.c_ulong if unsigned else ctypes.c_long
    if longs == 2:
        return ctypes.c_ulonglong if unsigned else ctypes.c_longlong
    return ctypes.c_uint if unsigned else ctypes.c_int


class Unsupported(Exception):
    """A declaration cannot be converted. `kind` is "unsupported" when the
    construct itself is the reason, "skipped-dependency" when something it
    uses was skipped."""

    def __init__(self, message: str, kind: DiagnosticKind = "unsupported", cause: tuple[str, str] | None = None):
        super().__init__(message)
        self.message = message
        self.kind: DiagnosticKind = kind
        # (name, reason) of the declaration skipped first, for a dependency
        self.cause = cause

    def cause_for(self, name: str) -> tuple[str, str]:
        """The root cause to record when `name` fails because of this."""
        return self.cause or (name, self.message)


def _skipped(name: str, cause: tuple[str, str]) -> Unsupported:
    """The error for using `name`, skipped because of `cause`, never
    nesting one "was skipped" in another."""
    root, why = cause
    return Unsupported(f"{name} was skipped ({why if root == name else f'{root}: {why}'})", "skipped-dependency", cause)


class _Bad:
    """A typedef name that cannot be used, with the reason."""

    def __init__(self, message: str, cause: tuple[str, str]):
        self.message = message
        self.cause = cause


class _FuncType:
    """A C function type (not a pointer to one): its CFUNCTYPE class."""

    def __init__(self, cls: type):
        self.cls = cls


class _Int:
    """An integer constant with its C type: width in bits and signedness."""

    __slots__ = ("bits", "unsigned", "value")

    def __init__(self, value: int, bits: int = _INT_BITS, unsigned: bool = False):
        bits = max(bits, _INT_BITS)  # integer promotion
        value &= (1 << bits) - 1
        if not unsigned and value >> (bits - 1):
            value -= 1 << bits
        self.value, self.bits, self.unsigned = value, bits, unsigned


def _typed_int(value: int) -> _Int:
    """An int with the narrowest type from int, long long and unsigned long long."""
    if -(1 << 31) <= value < 1 << 31:
        return _Int(value)
    return _Int(value, 64, value >= 1 << 63)


def _int_literal(text: str) -> _Int:
    body = text.rstrip("uUlL")
    suffix = text[len(body) :].lower()
    if body[:2] in ("0x", "0X", "0b", "0B"):
        value, decimal = int(body, 0), False
    elif len(body) > 1 and body[0] == "0":
        value, decimal = int(body, 8), False
    else:
        value, decimal = int(body), True
    widths = {0: [_INT_BITS, _LONG_BITS, 64], 1: [_LONG_BITS, 64], 2: [64]}[suffix.count("l")]
    for bits in widths:
        for unsigned in (False, True):
            if ("u" in suffix and not unsigned) or (unsigned and decimal and "u" not in suffix):
                continue
            if value < 1 << (bits - (0 if unsigned else 1)):
                return _Int(value, bits, unsigned)
    return _Int(value, 64, True)


_ESCAPES = {"n": "\n", "t": "\t", "r": "\r", "a": "\a", "b": "\b", "f": "\f", "v": "\v", "e": "\x1b"}
_ESCAPE = re.compile(r"\\(x[0-9a-fA-F]+|[0-7]{1,3}|u[0-9a-fA-F]{4}|U[0-9a-fA-F]{8}|.)", re.S)


def _unescape(text: str) -> str:
    def sub(m: re.Match[str]) -> str:
        e = m.group(1)
        if e[0] in "xuU":
            return chr(int(e[1:], 16))
        if e[0] in "01234567":
            return chr(int(e, 8))
        return _ESCAPES.get(e, e)

    return _ESCAPE.sub(sub, text)


def _literal_body(text: str, quote: str) -> str:
    return text[text.index(quote) + 1 : text.rindex(quote)]


def _pragma_text(node: Any) -> str:
    """The text of `#pragma text` or `_Pragma("text")`, which pycparser
    keeps as a string constant."""
    text = node.string
    if isinstance(text, c_ast.Constant):
        return _unescape(_literal_body(text.value, '"'))
    return str(text)


def _coord(node: Any) -> Any:
    """The first coordinate found on `node` or down its type chain."""
    while node is not None:
        if getattr(node, "coord", None) is not None:
            return node.coord
        node = getattr(node, "type", None)
    return None


def _definitions(node: Any) -> list[Any]:
    """The struct, union and enum definitions on the type chain of a declaration."""
    found = []
    while node is not None:
        if isinstance(node, (c_ast.Struct, c_ast.Union)) and node.decls is not None:
            found.append(node)
        elif isinstance(node, c_ast.Enum) and node.values is not None:
            found.append(node)
        node = getattr(node, "type", None)
    return found


class Converter:
    """Converts a FileAST into `ns`, name -> ctypes type, CFUNCTYPE class,
    int, float or str, in source order. `prelude` holds the names the
    prelude defined, `skipped` the skip diagnostic for each skipped name
    and `failed` the aggregate classes that could not be completed."""

    def __init__(
        self,
        rw: Rewritten,
        table: list[tuple[str, int] | None],
        tu_path: str,
        module: str,
        types: dict[str, type],
        recovered: set[str],
        lost: set[str],
        strict: bool,
        diags: list[Diagnostic],
    ):
        self.rw, self.table, self.tu_path, self.module = rw, table, tu_path, module
        self.types, self.recovered, self.strict, self.diags = types, recovered, strict, diags
        # classes whose definition was taken out after a parse error
        self.lost = lost
        self.ns: dict[str, object] = {}
        self.prelude: set[str] = set()
        self.skipped: dict[str, Diagnostic] = {}
        self.tags: dict[str, type] = {}
        self.anon: dict[int, type] = {}
        self.anon_names: dict[int, str] = {}
        self.definitions: dict[type, Any] = {}
        self.completed: set[type] = set()
        self.failed: dict[type, tuple[str, str]] = {}
        self.failed_enums: dict[str, tuple[str, str]] = {}
        self.typedefs: dict[str, object] = {}
        self.enums: dict[str, int] = {}
        self.pack: list[int | None] = [None]
        self.counter = 0
        # > 0 while evaluating the branch of a conditional that is not taken
        self.dead = 0
        self.in_prelude = False
        self.coord: Any = None
        self.field: str | None = None
        self.field_attrs: dict[int, list[Attr]] = {}
        self.span_attrs: dict[int, Attr] = {}
        for at in rw.attrs:
            if at.kind == "unsupported":
                index = rw.span_index(at.offset)
                if index is not None:
                    self.span_attrs.setdefault(index, at)
            else:
                self.field_attrs.setdefault(at.body, []).append(at)

    # -- bookkeeping -----------------------------------------------------------

    def source(self, coord: Any) -> tuple[str | None, int | None]:
        if coord is None or coord.line >= len(self.table):
            return None, None
        entry = self.table[coord.line]
        return entry if entry else (None, None)

    def diag(self, severity: Severity, kind: DiagnosticKind, name: str | None, message: str, coord: Any = None) -> None:
        if self.in_prelude:
            return
        if severity == "skip" and name is not None and name in self.skipped:
            return
        file, line = self.source(coord if coord is not None else self.coord)
        d = Diagnostic(severity, kind, name, file, line, message)
        if severity == "skip" and self.strict:
            raise HeaderError(f"{file}:{line}: {name + ': ' if name else ''}{message}")
        self.diags.append(d)
        if severity == "skip" and name is not None:
            self.skipped[name] = d

    def bind(self, name: str, value: object) -> None:
        self.ns[name] = value
        if self.in_prelude:
            self.prelude.add(name)
        else:
            self.prelude.discard(name)

    def new_class(self, name: str, base: type, attrs: dict[str, object] | None = None) -> type:
        return type(name, (base,), {"__module__": self.module, "__qualname__": name, **(attrs or {})})

    # -- types -----------------------------------------------------------------

    def type_of(self, node: Any, by_value: bool) -> Any:
        """The ctypes type of a pycparser type node; None for void, a
        `_FuncType` for a function type. `by_value` requires a struct or
        union to be complete."""
        if isinstance(node, (c_ast.TypeDecl, c_ast.Typename)):
            return self.type_of(node.type, by_value)
        if isinstance(node, c_ast.IdentifierType):
            return self.named(node.names, by_value)
        if isinstance(node, (c_ast.Struct, c_ast.Union)):
            return self.aggregate(node, by_value)
        if isinstance(node, c_ast.Enum):
            return self.enum(node)
        if isinstance(node, c_ast.PtrDecl):
            return self.pointer(node)
        if isinstance(node, c_ast.ArrayDecl):
            elem = self.type_of(node.type, True)
            if elem is None or isinstance(elem, _FuncType):
                raise Unsupported("array of void or of functions")
            n = 0 if node.dim is None else self.int_const(node.dim)
            if n < 0:
                raise Unsupported(f"array size {n} is negative")
            return elem * n
        if isinstance(node, c_ast.FuncDecl):
            return self.function(node)
        raise Unsupported(f"{type(node).__name__} is not supported")

    def named(self, names: list[str], by_value: bool) -> Any:
        if len(names) == 1:
            n = names[0]
            if n in self.types:
                return self.types[n]
            if n in STD_TYPES:
                return STD_TYPES[n]
            if n in self.typedefs:
                t = self.typedefs[n]
                if isinstance(t, _Bad):
                    raise Unsupported(t.message, "skipped-dependency", t.cause)
                if by_value:
                    self.require_complete(t, n)
                return t
        if "__int128" in names:
            raise Unsupported("__int128 has no ctypes type")
        return _base_type(names)

    def require_complete(self, t: object, name: str) -> None:
        if isinstance(t, type) and issubclass(t, _AGGREGATE) and t not in self.completed:
            if t in self.failed:
                raise _skipped(name, self.failed[t])
            if t.__name__ in self.lost:
                raise _skipped(name, (t.__name__, "its definition could not be parsed"))
            raise Unsupported(f"{name} is opaque (only forward declared) and used by value")

    def pointer(self, node: Any) -> type:
        try:
            t = self.type_of(node.type, False)
        except Unsupported as e:
            self.diag("note", "unsupported", self.field, f"pointer to an unsupported type is c_void_p: {e.message}")
            return ctypes.c_void_p
        if t is None:
            return ctypes.c_void_p
        if isinstance(t, _FuncType):
            return t.cls
        if t is ctypes.c_char:
            return ctypes.c_char_p
        if t is ctypes.c_wchar:
            return ctypes.c_wchar_p
        return ctypes.POINTER(t)

    def function(self, node: Any) -> _FuncType:
        args = []
        params = node.args.params if node.args is not None else []
        for p in params:
            if isinstance(p, c_ast.EllipsisParam):
                raise Unsupported("variadic function")
            pt = p.type
            # Array and function parameters decay to pointers.
            if isinstance(pt, c_ast.ArrayDecl):
                pt = c_ast.PtrDecl([], pt.type)
            elif isinstance(pt, c_ast.FuncDecl):
                pt = c_ast.PtrDecl([], pt)
            t = self.type_of(pt, True)
            if t is None:
                if len(params) == 1:
                    break
                raise Unsupported("void parameter")
            if isinstance(t, _FuncType):
                t = t.cls
            elif isinstance(t, type) and issubclass(t, ctypes.Array):
                t = ctypes.POINTER(getattr(t, "_type_"))
            args.append(t)
        res = self.type_of(node.type, True)
        if isinstance(res, _FuncType) or (isinstance(res, type) and issubclass(res, ctypes.Array)):
            raise Unsupported("function returning an array or a function")
        try:
            return _FuncType(ctypes.CFUNCTYPE(res, *args))
        except (TypeError, ValueError) as e:
            raise Unsupported(f"ctypes rejects the signature: {e}") from None

    # -- structs and unions ----------------------------------------------------

    def aggregate_class(self, node: Any) -> type:
        """The class of a struct or union node, created on first sight."""
        kind = "struct" if isinstance(node, c_ast.Struct) else "union"
        key = f"{kind} {node.name}" if node.name else None
        cls = self.tags.get(key) if key else self.anon.get(id(node))
        if cls is None:
            base = ctypes.Structure if kind == "struct" else ctypes.Union
            if key:
                cls = self.new_class(f"{kind}_{node.name}", base)
                self.tags[key] = cls
                self.bind(cls.__name__, cls)
            else:
                name = self.anon_names.get(id(node))
                if name is None:
                    self.counter += 1
                    name = f"anon_{kind}{self.counter}"
                cls = self.new_class(name, base)
                self.anon[id(node)] = cls
        return cls

    def aggregate(self, node: Any, by_value: bool) -> type:
        cls = self.aggregate_class(node)
        if node.decls is not None:
            first = self.definitions.get(cls)
            if first is None:
                self.definitions[cls] = node
                self.complete(cls, node)
            elif first is not node:
                self.redefinition(cls, node, first)
        if by_value:
            self.require_complete(cls, cls.__name__)
        return cls

    def redefinition(self, cls: type, node: Any, first: Any) -> None:
        # A header without include guards, included twice, is fine when the
        # definitions agree. A node's repr has everything but coordinates.
        if repr(node) != repr(first):
            self.failed.setdefault(cls, (cls.__name__, "conflicting definitions"))
            self.diag("skip", "unsupported", cls.__name__, f"{cls.__name__} is defined twice, differently", node.coord)

    def complete(self, cls: type, node: Any) -> None:
        saved = self.field
        try:
            fields, anonymous, pack, align = self.fields(node)
        except Unsupported as e:
            self.failed[cls] = e.cause_for(cls.__name__)
            if node.name is None and id(node) not in self.anon_names:
                raise  # a nested anonymous struct: its container reports it
            self.diag("skip", e.kind, cls.__name__, e.message, node.coord)
            return
        finally:
            self.field = saved
        try:
            if pack:
                setattr(cls, "_layout_", "ms")
                setattr(cls, "_pack_", pack)
            if align:
                setattr(cls, "_align_", align)
            if anonymous:
                setattr(cls, "_anonymous_", tuple(anonymous))
            setattr(cls, "_fields_", fields)
        except (TypeError, ValueError, AttributeError) as e:
            self.failed[cls] = (cls.__name__, f"ctypes rejects the layout: {e}")
            self.diag("skip", "unsupported", cls.__name__, self.failed[cls][1], node.coord)
            return
        self.completed.add(cls)

    def fields(self, node: Any) -> tuple[list[tuple[Any, ...]], list[str], int | None, int]:
        decls = list(node.decls)
        pack = self.pack[-1]
        align = 0
        offsets = {id(d): self.rw.offset(d.coord.line, d.coord.column) for d in decls if d.coord is not None}
        owners = self.owners(decls, offsets)
        real = {d.name for d in decls if getattr(d, "name", None)}
        counters = {"_jita_bits": 0, "_jita_anon": 0}

        def generated(prefix: str) -> str:
            while (n := f"{prefix}{counters[prefix]}") in real:
                counters[prefix] += 1
            counters[prefix] += 1
            return n

        fields: list[tuple[Any, ...]] = []
        anonymous: list[str] = []
        bits = False
        for d in decls:
            if isinstance(d, c_ast.Pragma):
                if _pragma_text(d).strip().startswith("pack"):
                    raise Unsupported("#pragma pack inside a struct body")
                continue
            if not isinstance(d, c_ast.Decl):
                continue
            name = d.name
            if name and name.startswith(MARKER):
                attr = name[len(MARKER) :]
                if attr == "packed":
                    pack = 1
                else:
                    align = max(align, int(attr.split("_")[1]))
                continue
            if name is None and d.bitsize is None and isinstance(d.type, (c_ast.Struct, c_ast.Union, c_ast.Enum)):
                if d.type.name is not None or isinstance(d.type, c_ast.Enum):
                    # A tag declared inside the body, not a member (C11).
                    self.field = None
                    self.type_of(d.type, False)
                    continue
            if name is None and d.bitsize is None and not isinstance(d.type, (c_ast.Struct, c_ast.Union)):
                # `struct S { Inner; }`: GCC ignores it, MSVC embeds Inner.
                raise Unsupported("an unnamed member that is not a struct or union (a Microsoft extension)")
            self.field = name
            try:
                t = self.type_of(d.type, True)
                if t is None or isinstance(t, _FuncType):
                    raise Unsupported("a field of void or function type")
                self.check_alignment(d, t, pack, owners.get(id(d), []))
                if d.bitsize is not None:
                    width = self.int_const(d.bitsize)
                    if width == 0:
                        raise Unsupported("zero width bit field")
                    if getattr(t, "_type_", None) not in (*_INT_CODES, "?"):
                        raise Unsupported(f"bit field of type {t.__name__}")
                    fields.append((name or generated("_jita_bits"), t, width))
                    bits = True
                elif name is None:
                    anon = generated("_jita_anon")
                    anonymous.append(anon)
                    fields.append((anon, t))
                else:
                    fields.append((name, t))
            except Unsupported as e:
                raise Unsupported(f"field {name or '(anonymous)'}: {e.message}", e.kind, e.cause) from None
        if pack and bits:
            raise Unsupported("bit fields in a packed struct (GCC and MSVC pack them differently)")
        return fields, anonymous, pack, align

    def owners(self, decls: list[Any], offsets: dict[int, int]) -> dict[int, list[Attr]]:
        """Field level attributes by the id of the member they apply to: the
        declarator right before the attribute, or every declarator of the
        member declaration when the attribute precedes them all."""
        first = next((offsets[id(d)] for d in decls if id(d) in offsets), None)
        if first is None:
            return {}
        result: dict[int, list[Attr]] = {}
        for at in self.field_attrs.get(self.rw.body_of(first), []):
            members = [d for d in decls if id(d) in offsets and at.start <= offsets[id(d)] < at.end]
            before = [d for d in members if offsets[id(d)] < at.offset]
            for d in before[-1:] or members:
                result.setdefault(id(d), []).append(at)
        return result

    def check_alignment(self, d: Any, t: type, pack: int | None, attrs: list[Attr]) -> None:
        """ctypes has no per field alignment: an explicit alignment is fine
        only when it does not exceed what the field gets anyway."""
        natural = ctypes.alignment(t)
        effective = min(natural, pack) if pack else natural
        want = 0
        for al in d.align or []:
            a = al.alignment
            if isinstance(a, c_ast.Typename):
                at = self.type_of(a, True)
                want = max(want, ctypes.alignment(at) if at is not None else 1)
            else:
                want = max(want, self.int_const(a))
        for at in attrs:
            if at.kind == "packed" and effective > 1:
                raise Unsupported(f"packed field {d.name}")
            if at.kind == "aligned":
                want = max(want, at.value)
        if want > effective:
            raise Unsupported(f"alignment {want} exceeds the natural alignment {effective}")

    # -- enums -----------------------------------------------------------------

    def enum(self, node: Any) -> type:
        key = f"enum {node.name}" if node.name else None
        if key and key in self.failed_enums:
            raise _skipped(f"enum_{node.name}", self.failed_enums[key])
        cls = self.tags.get(key) if key else self.anon.get(id(node))
        if cls is not None or node.values is None:
            return cls or ctypes.c_int
        members: dict[str, int] = {}
        value = -1
        for e in node.values.enumerators:
            value = self.int_const(e.value) if e.value is not None else value + 1
            members[e.name] = value
            self.enums[e.name] = value
        lo, hi = min(members.values(), default=0), max(members.values(), default=0)
        if -(1 << 31) <= lo and hi < 1 << 31:
            base: type = ctypes.c_int
        elif lo >= 0 and hi < 1 << 32:
            base = ctypes.c_uint
        else:
            base = ctypes.c_int64 if lo < 0 else ctypes.c_uint64
        if key:
            name = f"enum_{node.name}"
        else:
            name = self.anon_names.get(id(node)) or ""
            if not name:
                self.counter += 1
                name = f"anon_enum{self.counter}"
        attrs: dict[str, object] = {}
        for member, v in members.items():
            if member in _C_INT_ATTRS or member.startswith("__"):
                self.diag(
                    "note",
                    "shadowed-member",
                    member,
                    f"clashes with a ctypes attribute of {name}, top level only",
                    node.coord,
                )
            else:
                attrs[member] = v
        cls = self.new_class(name, base, attrs)
        if key:
            self.tags[key] = cls
            self.bind(name, cls)
        else:
            self.anon[id(node)] = cls
        for member, v in members.items():
            self.bind(member, v)
        return cls

    # -- constants -------------------------------------------------------------

    def int_const(self, node: Any) -> int:
        v = self.const(node)
        if not isinstance(v, _Int):
            raise Unsupported("not an integer constant expression")
        return v.value

    def const(self, node: Any) -> _Int | float | str:
        """Evaluate a constant expression with C semantics."""
        if isinstance(node, c_ast.Constant):
            t = node.type
            if t == "string":
                return _unescape(_literal_body(node.value, '"'))
            if t == "char":
                s = _unescape(_literal_body(node.value, "'"))
                if len(s) != 1:
                    raise Unsupported("multi-character constant")
                v = ord(s)
                # A plain char constant has the value of a char converted to int.
                signed_char = node.value[0] == "'" and not prelude.CHAR_UNSIGNED
                return _Int(v - 256 if signed_char and 127 < v < 256 else v)
            if t.endswith("int"):
                return _int_literal(node.value)
            if t in ("float", "double", "long double"):
                text = node.value.rstrip("fFlL")
                return float.fromhex(text) if text[:2].lower() == "0x" else float(text)
            raise Unsupported(f"{t} constant")
        if isinstance(node, c_ast.ID):
            if node.name in self.enums:
                return _typed_int(self.enums[node.name])
            # Macros are expanded by the preprocessor, so any other name is
            # not a constant.
            raise Unsupported(f"{node.name} is not a constant")
        if isinstance(node, c_ast.UnaryOp):
            return self.unary(node.op, node.expr)
        if isinstance(node, c_ast.BinaryOp):
            return self.binary(node.op, self.const(node.left), self.const(node.right))
        if isinstance(node, c_ast.TernaryOp):
            cond = self.const(node.cond)
            taken = bool(cond.value if isinstance(cond, _Int) else cond)
            # The branch not taken only contributes its type: division by
            # zero or a negative shift there is not an error.
            self.dead += 1
            try:
                other = self.const(node.iffalse if taken else node.iftrue)
            finally:
                self.dead -= 1
            chosen = self.const(node.iftrue if taken else node.iffalse)
            a, b = (chosen, other) if taken else (other, chosen)
            if isinstance(a, _Int) and isinstance(b, _Int):
                # The result has the common type of both branches.
                bits, unsigned = _common(a, b)
                return _Int(chosen.value if isinstance(chosen, _Int) else 0, bits, unsigned)
            if isinstance(a, str) or isinstance(b, str):
                raise Unsupported("a string in a conditional expression")
            return float(chosen.value if isinstance(chosen, _Int) else chosen)
        if isinstance(node, c_ast.Cast):
            return self.cast(node.to_type, self.const(node.expr))
        raise Unsupported(f"{type(node).__name__} in a constant expression")

    def unary(self, op: str, expr: Any) -> _Int | float | str:
        if op in ("sizeof", "_Alignof", "&", "*"):
            raise Unsupported(f"{op} in a constant expression")
        a = self.const(expr)
        if isinstance(a, str):
            raise Unsupported(f"{op} on a string")
        if not isinstance(a, _Int):
            if op == "-":
                return -a
            if op == "+":
                return a
            if op == "!":
                return _Int(int(not a))
            raise Unsupported(f"{op} on a float")
        if op == "-":
            return _Int(-a.value, a.bits, a.unsigned)
        if op == "+":
            return a
        if op == "~":
            return _Int(~a.value, a.bits, a.unsigned)
        if op == "!":
            return _Int(int(not a.value))
        raise Unsupported(f"{op} in a constant expression")

    def binary(self, op: str, a: _Int | float | str, b: _Int | float | str) -> _Int | float:
        if isinstance(a, str) or isinstance(b, str):
            raise Unsupported("arithmetic on a string")
        if op in ("&&", "||"):
            x = a.value if isinstance(a, _Int) else a
            y = b.value if isinstance(b, _Int) else b
            return _Int(int(bool(x and y) if op == "&&" else bool(x or y)))
        if not (isinstance(a, _Int) and isinstance(b, _Int)):
            x = a.value if isinstance(a, _Int) else a
            y = b.value if isinstance(b, _Int) else b
            match op:
                case "+":
                    return x + y
                case "-":
                    return x - y
                case "*":
                    return x * y
                case "/" if y:
                    return x / y
                case "/" if self.dead:
                    return 0.0
                case "<" | ">" | "<=" | ">=" | "==" | "!=":
                    return _Int(int(_compare(op, x, y)))
            raise Unsupported(f"{op} on a float")
        if op in ("<<", ">>"):
            if b.value < 0:
                if self.dead:
                    return _Int(0, a.bits, a.unsigned)
                raise Unsupported("negative shift count")
            return _Int(a.value << b.value if op == "<<" else a.value >> b.value, a.bits, a.unsigned)
        bits, unsigned = _common(a, b)
        x, y = _Int(a.value, bits, unsigned).value, _Int(b.value, bits, unsigned).value
        if op in ("<", ">", "<=", ">=", "==", "!="):
            return _Int(int(_compare(op, x, y)))
        if op in ("/", "%"):
            if y == 0:
                if self.dead:
                    return _Int(0, bits, unsigned)
                raise Unsupported("division by zero")
            q = abs(x) // abs(y) * (1 if (x < 0) == (y < 0) else -1)
            return _Int(q if op == "/" else x - q * y, bits, unsigned)
        match op:
            case "+":
                r = x + y
            case "-":
                r = x - y
            case "*":
                r = x * y
            case "&":
                r = x & y
            case "|":
                r = x | y
            case "^":
                r = x ^ y
            case _:
                raise Unsupported(f"{op} in a constant expression")
        return _Int(r, bits, unsigned)

    def cast(self, to_type: Any, v: _Int | float | str) -> _Int | float:
        t = self.type_of(to_type, True)
        code = getattr(t, "_type_", None)
        if isinstance(v, str) or not isinstance(code, str):
            raise Unsupported("cast to a non arithmetic type")
        x = v.value if isinstance(v, _Int) else v
        if code in "fdg":
            return float(x)
        if code == "?":
            return _Int(int(bool(x)))
        if code in _INT_CODES or code == "c":
            bits = 8 * ctypes.sizeof(t)
            unsigned = code in "BHILQ" or (code == "c" and prelude.CHAR_UNSIGNED)
            n = _Int(int(x), bits, unsigned)
            if bits < _INT_BITS:
                # wrap to the narrow type, then promote to int
                n = _Int(int(x) & ((1 << bits) - 1), _INT_BITS, False)
                if not unsigned and n.value >> (bits - 1):
                    n = _Int(n.value - (1 << bits))
            return n
        raise Unsupported("cast to a non arithmetic type")

    # -- declarations ----------------------------------------------------------

    def run(self, ast: Any) -> None:
        for ext in ast.ext:
            if isinstance(ext, c_ast.Typedef) and isinstance(ext.type, c_ast.TypeDecl):
                inner = ext.type.type
                if isinstance(inner, (c_ast.Struct, c_ast.Union, c_ast.Enum)) and inner.name is None:
                    self.anon_names.setdefault(id(inner), ext.name)
        for ext in ast.ext:
            if isinstance(ext, c_ast.Pragma):
                self.pragma(_pragma_text(ext))
                continue
            if not isinstance(ext, (c_ast.Typedef, c_ast.Decl)):
                continue
            self.coord = _coord(ext)
            file, _ = self.source(self.coord)
            self.in_prelude = file == self.tu_path
            self.field = getattr(ext, "name", None)
            try:
                offset = self.rw.offset(self.coord.line, self.coord.column or 1) if self.coord else -1
                index = self.rw.span_index(offset)
                if index is not None and index in self.span_attrs:
                    raise Unsupported(f"attribute {self.span_attrs[index].text} is not supported")
                if isinstance(ext, c_ast.Typedef):
                    self.typedef(ext)
                else:
                    self.decl(ext)
            except Unsupported as e:
                self.declaration_failed(ext, e)
        self.in_prelude = False

    def declaration_failed(self, ext: Any, e: Unsupported) -> None:
        name: str | None = ext.name
        bare = isinstance(ext.type, (c_ast.Struct, c_ast.Union, c_ast.Enum)) and not _definitions(ext.type)
        if name is None and bare and e.message.startswith("attribute "):
            # GCC ignores attributes on a declaration without a body.
            self.diag("note", "unsupported", None, f"{e.message}, ignored on a declaration without a body")
            return
        reported = False
        for node in _definitions(ext.type):
            if isinstance(node, c_ast.Enum):
                if node.name:
                    tag = f"enum_{node.name}"
                    self.failed_enums[f"enum {node.name}"] = e.cause_for(tag)
                    name = name or tag
                continue
            cls = self.aggregate_class(node)
            self.definitions.setdefault(cls, node)
            if cls not in self.completed:
                self.failed.setdefault(cls, e.cause_for(cls.__name__))
                if node.name or id(node) in self.anon_names:
                    self.diag("skip", e.kind, cls.__name__, e.message)
                    reported = True
        if isinstance(ext, c_ast.Typedef):
            cause = e.cause_for(ext.name)
            self.typedefs[ext.name] = _Bad(_skipped(ext.name, cause).message, cause)
        if name is not None or not reported:
            self.diag("skip", e.kind, name, e.message)

    def typedef(self, ext: Any) -> None:
        name = ext.name
        if name in self.types:
            self.typedefs[name] = self.types[name]
            self.bind(name, self.types[name])
            return
        if name in STD_TYPES:
            self.typedefs[name] = STD_TYPES[name]
            self.bind(name, STD_TYPES[name])
            return
        target = ext.type
        if (
            isinstance(target, c_ast.TypeDecl)
            and isinstance(target.type, c_ast.IdentifierType)
            and target.type.names == [UNKNOWN]
        ):
            if name in self.recovered and not self.in_prelude:
                why = "its declaration could not be parsed"
                self.typedefs[name] = _Bad(f"{name} was skipped ({why})", (name, why))
            else:
                why = f"{name} is a system type of unknown size, pass types={{{name!r}: <ctypes type>}} to load()"
                self.typedefs[name] = _Bad(why, (name, why))
            return
        t = self.type_of(target, False)
        if isinstance(t, type) and t in self.failed:
            raise _skipped(t.__name__, self.failed[t])
        self.typedefs[name] = t
        self.bind(name, t.cls if isinstance(t, _FuncType) else t)

    def decl(self, ext: Any) -> None:
        t = ext.type
        if isinstance(t, (c_ast.Struct, c_ast.Union, c_ast.Enum)):
            self.field = None
            self.type_of(t, False)
        elif isinstance(t, c_ast.FuncDecl):
            self.bind(ext.name, self.function(t).cls)
        else:
            # An object declaration binds nothing, but may define tags.
            self.field = None
            for node in _definitions(t):
                self.type_of(node, False)

    def pragma(self, text: str) -> None:
        m = re.fullmatch(r"\s*pack\s*\((.*)\)\s*", text)
        if not m:
            return
        args = [a.strip() for a in m.group(1).split(",") if a.strip()]
        numbers = [int(a, 0) for a in args if a[:1].isdigit()]
        if not args:
            self.pack[-1] = None
        elif args[0] == "push":
            self.pack.append(numbers[0] if numbers else self.pack[-1])
        elif args[0] == "pop":
            if len(self.pack) > 1:
                self.pack.pop()
        elif numbers:
            self.pack[-1] = numbers[0]

    # -- macros ----------------------------------------------------------------

    def evaluate_macros(self, macros: list[tuple[str, str, str, int]]) -> None:
        """Bind the object like macros whose expansion is a constant."""
        parser = pycparser.CParser()
        type_names = set(self.typedefs) | set(STD_TYPES) | set(self.types)
        for name, text, file, line in macros:
            if not text:
                continue
            if len(name) > 4 and name.startswith("__") and name.endswith("__"):
                continue
            if name in self.ns and name not in self.prelude:
                self.diags.append(
                    Diagnostic(
                        "note", "shadowed-macro", name, file, line, f"a declaration named {name} takes precedence"
                    )
                )
                continue
            words = set(re.findall(r"[A-Za-z_]\w*", text)) & type_names
            prefix = "".join(f"typedef int {w};" for w in sorted(words))
            try:
                ast = parser.parse(f"{prefix}\nenum {{ __jita_x = ({text}) }};", "<macro>")
                value = self.const(ast.ext[-1].type.values.enumerators[0].value)
            except (pycparser.c_parser.ParseError, Unsupported, ValueError, OverflowError, TypeError):
                continue
            self.bind(name, value.value if isinstance(value, _Int) else value)


def _common(a: _Int, b: _Int) -> tuple[int, bool]:
    """The width and signedness of the usual arithmetic conversions."""
    if a.unsigned == b.unsigned:
        return max(a.bits, b.bits), a.unsigned
    s, u = (b, a) if a.unsigned else (a, b)
    return (u.bits, True) if u.bits >= s.bits else (s.bits, False)


def _compare(op: str, x: float, y: float) -> bool:
    match op:
        case "<":
            return x < y
        case ">":
            return x > y
        case "<=":
            return x <= y
        case ">=":
            return x >= y
        case "==":
            return x == y
    return x != y
