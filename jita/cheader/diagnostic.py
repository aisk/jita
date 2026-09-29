"""What `jita.cheader.load` records about the declarations it skipped or
changed. Standard library only, so `jita.cheader` imports without the
optional dependencies."""

from dataclasses import dataclass
from typing import Literal

__all__ = ["Diagnostic", "DiagnosticKind", "Severity"]

type Severity = Literal["note", "skip"]
type DiagnosticKind = Literal[
    "missing-include",
    "preprocessor",
    "parse-error",
    "unsupported",
    "skipped-dependency",
    "shadowed-macro",
    "shadowed-member",
]


@dataclass(frozen=True)
class Diagnostic:
    """One finding of `load`. A `skip` means a declaration did not make it
    into the namespace, a `note` that something was ignored or degraded
    without losing a declaration."""

    severity: Severity
    kind: DiagnosticKind
    name: str | None
    file: str | None
    line: int | None
    message: str

    def __str__(self) -> str:
        where = f"{self.file}:{self.line}: " if self.file else ""
        what = f"{self.name}: " if self.name else ""
        return f"{where}{self.severity}: {what}{self.message} [{self.kind}]"
