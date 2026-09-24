"""Exception hierarchy. Every error raised by jita derives from JitaError."""


class JitaError(Exception):
    """Base class for all jita errors."""


class EncodeError(JitaError):
    """An instruction or operand cannot be encoded."""


class LinkError(JitaError):
    """A patch cannot be resolved or its value does not fit."""


class LoadError(JitaError):
    """Executable memory cannot be allocated, written or protected."""
