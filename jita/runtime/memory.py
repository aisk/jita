"""Executable memory: mapped read-write, filled, then flipped to read-execute."""

from __future__ import annotations

import ctypes
import mmap
import sys
from collections.abc import Callable

from ..core.errors import LoadError

_libc = None


def _mprotect(addr: int, size: int, prot: int) -> None:
    global _libc
    if _libc is None:
        _libc = ctypes.CDLL(None, use_errno=True)
        _libc.mprotect.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int)
        _libc.mprotect.restype = ctypes.c_int
    if _libc.mprotect(addr, size, prot) != 0:
        err = ctypes.get_errno()
        raise LoadError(f"mprotect({addr:#x}, {size:#x}) failed: errno {err}")


class ExecMemory:
    """A page aligned anonymous mapping, RW until `protect_exec` makes it
    (or a page aligned prefix of it) RX."""

    def __init__(self, size: int, icache_flush: Callable[[int, int], None] | None = None):
        if sys.platform == "win32":
            raise NotImplementedError("jita: executable memory on Windows is not implemented yet")
        if size < 0:
            raise LoadError(f"cannot map a negative size ({size})")
        page = mmap.PAGESIZE
        self.size = max(page, -(-size // page) * page)
        self._icache_flush = icache_flush
        try:
            self._map: mmap.mmap | None = mmap.mmap(
                -1, self.size, prot=mmap.PROT_READ | mmap.PROT_WRITE
            )
        except (OSError, OverflowError) as e:
            raise LoadError(f"cannot map {self.size:#x} bytes: {e}") from e
        anchor = ctypes.c_char.from_buffer(self._map)
        self.address = ctypes.addressof(anchor)
        del anchor  # release the buffer export so the map can be closed
        self.executable = False
        self.exec_size = 0  # size of the RX prefix once executable

    @property
    def closed(self) -> bool:
        return self._map is None

    def write(self, data: bytes, offset: int = 0) -> None:
        """Copy `data` to `offset`. After `protect_exec` only the writable
        tail (offset >= exec_size) accepts writes."""
        if self._map is None:
            raise LoadError("memory is closed")
        if offset < 0 or offset + len(data) > self.size:
            raise LoadError(f"write of {len(data)} bytes at {offset:#x} exceeds {self.size:#x}")
        if self.executable and offset < self.exec_size:
            raise LoadError(
                f"write at {offset:#x} touches the executable prefix ({self.exec_size:#x} bytes)"
            )
        self._map[offset : offset + len(data)] = data

    def protect_exec(self, exec_size: int | None = None) -> None:
        """Make the first `exec_size` bytes (default: all) read-execute.
        The rest of the mapping stays read-write. `exec_size` must be a
        multiple of the page size."""
        if self._map is None:
            raise LoadError("memory is closed")
        if exec_size is None:
            exec_size = self.size
        if not 0 <= exec_size <= self.size or exec_size % mmap.PAGESIZE:
            raise LoadError(f"bad exec_size {exec_size:#x} for a {self.size:#x} byte mapping")
        if exec_size:
            _mprotect(self.address, exec_size, mmap.PROT_READ | mmap.PROT_EXEC)
        self.executable = True
        self.exec_size = exec_size
        if self._icache_flush is not None and exec_size:
            self._icache_flush(self.address, exec_size)

    def close(self) -> None:
        if self._map is not None:
            self._map.close()
            self._map = None

    def __enter__(self) -> ExecMemory:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
