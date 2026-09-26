import ctypes
import mmap
import platform
import signal
import subprocess
import sys

import pytest

import jita.x64 as x64
from jita import Assembler, Extern, Label, LinkError, LoadError
from jita.core import ABS64, REL32

pytestmark = pytest.mark.skipif(
    platform.machine().lower() not in ("x86_64", "amd64") or platform.system() == "Windows",
    reason="needs an x86-64 POSIX host",
)




def test_return_42():
    a = Assembler(x64)
    a.bytes(b"\xb8\x2a\x00\x00\x00\xc3")  # mov eax, 42; ret
    with a.load() as mod:
        assert mod.function(ctypes.c_int)() == 42


def test_add_two_ints():
    a = Assembler(x64)
    a.bytes(b"\x8d\x04\x37\xc3")  # lea eax, [rdi+rsi]; ret  (SysV)
    mod = a.load()
    f = mod.function(ctypes.c_int, ctypes.c_int, ctypes.c_int)
    del mod  # the function keeps the module alive
    assert f(40, 2) == 42


def test_jumps_and_entry_label():
    a = Assembler(x64)
    ret1, done = Label(), Label()
    a.bytes(b"\xb8\x01\x00\x00\x00")  # mov eax, 1
    a.bytes(b"\xe9")
    a.emit_patch(REL32, done)  # forward
    a.bind(ret1)
    a.bytes(b"\xb8\x07\x00\x00\x00")  # mov eax, 7
    a.bytes(b"\xc3")
    a.bind(done)
    a.bytes(b"\x83\xf8\x01")  # cmp eax, 1
    a.bytes(b"\x0f\x84")
    a.emit_patch(REL32, ret1)  # je ret1 (backward)
    a.bytes(b"\xc3")
    a.align(16)
    a.label("second")
    a.bytes(b"\xb8\x05\x00\x00\x00\xc3")  # mov eax, 5; ret
    with a.load() as mod:
        assert mod.function(ctypes.c_int)() == 7
        assert mod.function(ctypes.c_int, entry="second")() == 5
        assert mod.address("second") % 16 == 0


def test_rip_relative_data_section():
    a = Assembler(x64)
    with a.section("data"):
        a.qword(0x1111)
        val = a.label()
        a.qword(0xDEADBEEF12345678)
    a.bytes(b"\x48\x8b\x05")  # mov rax, [rip+disp32]
    a.emit_patch(REL32, val)
    a.bytes(b"\xc3")
    with a.load() as mod:
        assert mod.function(ctypes.c_uint64)() == 0xDEADBEEF12345678


def test_data_pointer_via_abs64():
    a = Assembler(x64)
    with a.section("data"):
        target = a.label()
        a.qword(99)
        slot = a.label()
        a.qword(target)  # pointer to `target`
    a.bytes(b"\x48\x8b\x05")  # mov rax, [rip+slot]
    a.emit_patch(REL32, slot)
    a.bytes(b"\x48\x8b\x00\xc3")  # mov rax, [rax]; ret
    with a.load() as mod:
        assert mod.function(ctypes.c_int64)() == 99


def test_call_extern():
    cb_type = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_int)
    cb = cb_type(lambda v: v * 3)
    cb_addr = ctypes.cast(cb, ctypes.c_void_p).value
    assert cb_addr is not None

    a = Assembler(x64)
    a.bytes(b"\x48\x83\xec\x08")  # sub rsp, 8 (align stack)
    a.bytes(b"\x48\xb8")  # mov rax, imm64
    a.emit_patch(ABS64, Extern("triple"))
    a.bytes(b"\xff\xd0")  # call rax
    a.bytes(b"\x48\x83\xc4\x08\xc3")  # add rsp, 8; ret
    with a.load(externs={"triple": cb_addr}) as mod:
        assert mod.function(ctypes.c_int, ctypes.c_int)(14) == 42


def test_load_failure_releases_memory():
    a = Assembler(x64)
    a.bytes(b"\xe9")
    a.emit_patch(REL32, Label("missing"))
    with pytest.raises(LinkError):
        a.load()


def _store_then_load(section_kwargs):
    """mov dword [rip+d], 1; mov eax, dword [rip+d]; ret"""
    a = Assembler(x64)
    with a:
        with a.section("data", **section_kwargs):
            d = a.label()
            a.dword(0)
        x64.mov(x64.dword[x64.rip + d], 1)
        x64.mov(x64.eax, x64.dword[x64.rip + d])
        x64.ret()
    return a


def test_writable_data_section():
    a = _store_then_load({"writable": True})
    assert a.sections["data"].writable
    with a.load() as mod:
        assert mod.function(ctypes.c_int)() == 1
        assert mod.function(ctypes.c_int)() == 1
        assert mod.image.exec_size == mod.image.page_size
        assert mod.address(a.labels[0]) - mod.image.base == mod.image.page_size


def test_writable_flag_is_kept_by_later_switches():
    a = Assembler(x64)
    a.section("data", writable=True)
    a.section("code")
    a.section("data")
    assert a.sections["data"].writable
    a.section("data", writable=False)
    assert not a.sections["data"].writable


def test_writable_layout():
    a = Assembler(x64)
    a.bytes(b"\x90" * 5000)
    with a.section("rw1", writable=True):
        a.bytes(b"\x01")
    with a.section("rodata"):
        a.bytes(b"\x02" * 3)
    with a.section("rw2", writable=True, align=64):
        a.bytes(b"\x03")
    img = a.link()
    page = img.page_size
    assert page == mmap.PAGESIZE
    # Read-only sections first, in creation order, then the writable ones.
    assert img.section_offsets["code"] == 0
    assert img.section_offsets["rodata"] == 5008
    assert img.exec_size == -(-5011 // page) * page
    assert img.section_offsets["rw1"] == img.exec_size
    assert img.section_offsets["rw1"] % page == 0
    assert img.section_offsets["rw2"] == img.exec_size + 64
    assert img.data[img.exec_size] == 1 and img.data[img.exec_size + 64] == 3
    assert len(img.data) == img.exec_size + 65


def test_read_only_layout_unchanged():
    a = Assembler(x64)
    a.bytes(b"\x90" * 3)
    with a.section("data"):
        a.bytes(b"\x01")
    img = a.link()
    assert img.section_offsets == {"code": 0, "data": 16}
    assert len(img.data) == 17
    assert img.exec_size == img.page_size


def test_non_writable_data_section_is_read_only():
    # Writing to it faults; run it in a child process.
    script = (
        "import ctypes, jita.x64 as x64\n"
        "from jita import Assembler\n"
        "a = Assembler(x64)\n"
        "with a:\n"
        "    with a.section('data'):\n"
        "        d = a.label(); a.dword(0)\n"
        "    x64.mov(x64.dword[x64.rip + d], 1)\n"
        "    x64.ret()\n"
        "a.load().function(None)()\n"
    )
    r = subprocess.run([sys.executable, "-c", script], capture_output=True)
    sigbus = getattr(signal, "SIGBUS", signal.SIGSEGV)  # not on Windows (this module is skipped there)
    assert r.returncode in (-signal.SIGSEGV, -sigbus), r.stderr.decode()



def test_image_address_rejects_foreign_label():
    a, b = Assembler(x64), Assembler(x64)
    a.bytes(b"\x90" * 8)
    b.bytes(b"\x90" * 4)
    foreign = b.label("x")
    img = a.link(0x1000)
    with pytest.raises(LinkError, match="different assembler"):
        img.address(foreign)
    with a.load() as mod:
        with pytest.raises(LinkError, match="different assembler"):
            mod.address(foreign)
        with pytest.raises(LinkError, match="different assembler"):
            mod.function(ctypes.c_int, entry=foreign)
    own = a.label()
    assert a.link(0x1000).address(own) == 0x1008


def test_link_base_alignment():
    a = Assembler(x64)
    a.bytes(b"\x90")
    a.link(16)
    with pytest.raises(LinkError, match="aligned"):
        a.link(8)
    with a.section("data", align=256):
        a.bytes(b"\x00")
    a.link(0x1000)
    with pytest.raises(LinkError, match="aligned"):
        a.link(0x1080)


def test_load_rejects_alignment_above_page_size():
    a = Assembler(x64)
    a.bytes(b"\xc3")
    with a.section("big", align=2 * mmap.PAGESIZE):
        a.bytes(b"\x00")
    with pytest.raises(LoadError, match="page size"):
        a.load()
    a.link(0)  # linking alone is fine


def test_closed_module_rejects_use():
    a = Assembler(x64)
    a.bytes(b"\xc3")
    lbl = a.label("end")
    mod = a.load()
    mod.close()
    for call in (
        lambda: mod.function(None),
        lambda: mod.address("end"),
        lambda: mod.address(lbl),
    ):
        with pytest.raises(LoadError, match="closed"):
            call()
    mod.close()  # double close is a no-op


def test_image_outlives_assembler():
    # Image only keeps weak references to sections, but a live label keeps
    # its section alive, so the lookup still works after the assembler is gone.
    import gc

    a = Assembler(x64)
    a.bytes(b"\x90" * 4)
    lbl = a.label("end")
    img = a.link(0x1000)
    del a
    gc.collect()
    assert img.address(lbl) == 0x1004
    assert img.address("end") == 0x1004



def test_only_writable_sections():
    a = Assembler(x64)
    with a.section("vars", writable=True):
        a.label("v")
        a.qword(5)
    with a.load() as mod:
        assert mod.image.exec_size == 0
        assert mod.image.section_offsets["vars"] == 0
        cell = ctypes.c_int64.from_address(mod.address("v"))
        assert cell.value == 5
        cell.value = 6
        assert cell.value == 6



def test_module_write_patches_writable_data():
    a = Assembler(x64)
    with a.section("vars", writable=True):
        v = a.label("v")
        a.qword(5)
    with a:
        x64.mov(x64.rax, x64.qword[x64.rip + v])
        x64.ret()
    with a.load() as mod:
        get = mod.function(ctypes.c_int64)
        assert get() == 5
        mod.write(v, (7).to_bytes(8, "little"))
        assert get() == 7
        mod.write("v", (-3).to_bytes(8, "little", signed=True))
        assert get() == -3
        mod.write(mod.image.section_offsets["vars"], (11).to_bytes(8, "little"))
        assert get() == 11
        with pytest.raises(LoadError, match="executable"):
            mod.write(0, b"\x90")
        with pytest.raises(LoadError, match="outside the image"):
            mod.write(len(mod.image.data), b"\x00")


def test_module_rejects_use_after_memory_close():
    a = Assembler(x64)
    a.bytes(b"\xc3")
    a.label("end")
    mod = a.load()
    mod.memory.close()
    for call in (
        lambda: mod.function(None),
        lambda: mod.address("end"),
        lambda: mod.write("end", b""),
    ):
        with pytest.raises(LoadError, match="closed"):
            call()
