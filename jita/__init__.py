"""jita: runtime code generation at the abstraction level of DynASM."""

from .core import (
    Assembler,
    EncodeError,
    Extern,
    Image,
    JitaError,
    Label,
    LinkError,
    LoadError,
    PcLabels,
    current,
    label,
)
from .runtime import (
    METH_FASTCALL,
    METH_KEYWORDS,
    METH_NOARGS,
    METH_O,
    METH_VARARGS,
    Module,
    builtin,
    function,
)

__all__ = [
    "METH_FASTCALL",
    "METH_KEYWORDS",
    "METH_NOARGS",
    "METH_O",
    "METH_VARARGS",
    "Assembler",
    "EncodeError",
    "Extern",
    "Image",
    "JitaError",
    "Label",
    "LinkError",
    "LoadError",
    "Module",
    "PcLabels",
    "builtin",
    "current",
    "function",
    "label",
]
