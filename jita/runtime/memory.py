"""Executable memory: mapped read-write, filled, then flipped to read-execute.

POSIX hosts use mmap and mprotect. Windows uses VirtualAlloc, VirtualProtect
and FlushInstructionCache from kernel32."""

import ctypes
import mmap
import sys
import weakref
from collections.abc import Callable
from typing import Self

from ..core.errors import LoadError

if sys.platform == "win32":
    _MEM_COMMIT = 0x1000
    _MEM_RESERVE = 0x2000
    _MEM_RELEASE = 0x8000
    _PAGE_READWRITE = 0x04
    _PAGE_EXECUTE_READ = 0x20
    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _kernel32.VirtualAlloc.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint32, ctypes.c_uint32)
    _kernel32.VirtualAlloc.restype = ctypes.c_void_p
    _kernel32.VirtualProtect.argtypes = (
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_uint32),
    )
    _kernel32.VirtualProtect.restype = ctypes.c_int
    _kernel32.VirtualFree.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint32)
    _kernel32.VirtualFree.restype = ctypes.c_int
    _kernel32.GetCurrentProcess.argtypes = ()
    _kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    _kernel32.FlushInstructionCache.argtypes = (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t)
    _kernel32.FlushInstructionCache.restype = ctypes.c_int

    def _virtual_alloc(size: int) -> int:
        try:
            addr = _kernel32.VirtualAlloc(None, size, _MEM_COMMIT | _MEM_RESERVE, _PAGE_READWRITE)
        except (ctypes.ArgumentError, OverflowError) as e:
            raise LoadError(f"cannot map {size:#x} bytes: {e}") from e
        if not addr:
            raise LoadError(f"cannot map {size:#x} bytes: error {ctypes.get_last_error()}")
        return addr

    def _virtual_free(addr: int) -> None:
        _kernel32.VirtualFree(addr, 0, _MEM_RELEASE)

    def _protect_exec(addr: int, size: int) -> None:
        old = ctypes.c_uint32()
        if not _kernel32.VirtualProtect(addr, size, _PAGE_EXECUTE_READ, ctypes.byref(old)):
            raise LoadError(
                f"VirtualProtect({addr:#x}, {size:#x}) failed: error {ctypes.get_last_error()}"
            )
        _kernel32.FlushInstructionCache(_kernel32.GetCurrentProcess(), addr, size)

else:
    _libc = None

    def _protect_exec(addr: int, size: int) -> None:
        global _libc
        if _libc is None:
            _libc = ctypes.CDLL(None, use_errno=True)
            _libc.mprotect.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int)
            _libc.mprotect.restype = ctypes.c_int
        if _libc.mprotect(addr, size, mmap.PROT_READ | mmap.PROT_EXEC) != 0:
            raise LoadError(f"mprotect({addr:#x}, {size:#x}) failed: errno {ctypes.get_errno()}")


class ExecMemory:
    """A page aligned anonymous mapping, RW until `protect_exec` makes it
    (or a page aligned prefix of it) RX. Closing it, or dropping the last
    reference, releases the memory."""

    def __init__(self, size: int, icache_flush: Callable[[int, int], None] | None = None):
        if size < 0:
            raise LoadError(f"cannot map a negative size ({size})")
        page = mmap.PAGESIZE
        self.size = max(page, -(-size // page) * page)
        self._icache_flush = icache_flush
        self._map: mmap.mmap | None = None
        if sys.platform == "win32":
            self.address = _virtual_alloc(self.size)
            self._free = weakref.finalize(self, _virtual_free, self.address)
        else:
            try:
                self._map = mmap.mmap(-1, self.size, prot=mmap.PROT_READ | mmap.PROT_WRITE)
            except (OSError, OverflowError) as e:
                raise LoadError(f"cannot map {self.size:#x} bytes: {e}") from e
            anchor = ctypes.c_char.from_buffer(self._map)
            self.address = ctypes.addressof(anchor)
            del anchor  # release the buffer export so the map can be closed
            self._free = weakref.finalize(self, self._map.close)
        self.executable = False
        self.exec_size = 0  # size of the RX prefix once executable

    @property
    def closed(self) -> bool:
        return not self._free.alive

    def write(self, data: bytes, offset: int = 0) -> None:
        """Copy `data` to `offset`. After `protect_exec` only the writable
        tail (offset >= exec_size) accepts writes."""
        if self.closed:
            raise LoadError("memory is closed")
        if offset < 0 or offset + len(data) > self.size:
            raise LoadError(f"write of {len(data)} bytes at {offset:#x} exceeds {self.size:#x}")
        if self.executable and offset < self.exec_size:
            raise LoadError(
                f"write at {offset:#x} touches the executable prefix ({self.exec_size:#x} bytes)"
            )
        ctypes.memmove(self.address + offset, data, len(data))

    def protect_exec(self, exec_size: int | None = None) -> None:
        """Make the first `exec_size` bytes (default: all) read-execute.
        The rest of the mapping stays read-write. `exec_size` must be a
        multiple of the page size."""
        if self.closed:
            raise LoadError("memory is closed")
        if exec_size is None:
            exec_size = self.size
        if not 0 <= exec_size <= self.size or exec_size % mmap.PAGESIZE:
            raise LoadError(f"bad exec_size {exec_size:#x} for a {self.size:#x} byte mapping")
        if exec_size:
            _protect_exec(self.address, exec_size)
        self.executable = True
        self.exec_size = exec_size
        if self._icache_flush is not None and exec_size:
            self._icache_flush(self.address, exec_size)

    def close(self) -> None:
        self._free()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
