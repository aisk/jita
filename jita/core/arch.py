"""The interface an architecture package exposes to the Assembler."""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .assembler import Assembler


class Arch:
    """Architecture description consumed by core.

    Arch packages expose a single instance as `<package>.ARCH`.
    """

    name: str = "abstract"
    pointer_size: int = 8

    @property
    def insns(self) -> dict[str, Callable[..., Any]]:
        """Mnemonic name -> function taking `(*operands, asm=None)`."""
        return {}

    @property
    def assembler_class(self) -> type[Assembler]:
        """The class `Assembler(arch)` instantiates for this architecture."""
        from .assembler import Assembler

        return Assembler

    def nop_fill(self, n: int) -> bytes:
        """Exactly `n` bytes of padding that is safe to execute."""
        raise NotImplementedError

    def icache_flush(self, addr: int, size: int) -> None:
        """Make freshly written code visible to instruction fetch."""

    def __repr__(self) -> str:
        return f"<Arch {self.name}>"
