"""Built in tables: the prelude every translation unit starts with, the
standard type names, the predefined macros of the host compiler and the
attribute whitelist.

The prelude replaces the system headers, which are not read. It is
adapted from the fake libc headers of pycparser (utils/fake_libc_include,
BSD license, see LICENSE): every system type is a typedef of
`__jita_unknown_t`, so the converter knows its size is unknown instead of
taking it for an int, and the limits macros are computed from ctypes.
"""

import ctypes
import platform
import sys

__all__ = [
    "CHAR_UNSIGNED",
    "HARMLESS_ATTRIBUTES",
    "LAYOUT_ATTRIBUTES",
    "PRELUDE",
    "STD_TYPES",
    "UNKNOWN",
    "predefined_macros",
]

UNKNOWN = "__jita_unknown_t"

# Names a header can use as types without including the system header
# that declares them. The ones in STD_TYPES are mapped by name first, the
# rest have no known layout.
_SYSTEM_TYPES = """
    size_t ssize_t ptrdiff_t wchar_t char16_t char32_t wint_t max_align_t
    int8_t uint8_t int16_t uint16_t int32_t uint32_t int64_t uint64_t
    int_least8_t uint_least8_t int_least16_t uint_least16_t
    int_least32_t uint_least32_t int_least64_t uint_least64_t
    int_fast8_t uint_fast8_t int_fast16_t uint_fast16_t
    int_fast32_t uint_fast32_t int_fast64_t uint_fast64_t
    intptr_t uintptr_t intmax_t uintmax_t
    __int8_t __uint8_t __int16_t __uint16_t __int32_t __uint32_t __int64_t __uint64_t
    __s8 __u8 __s16 __u16 __s32 __u32 __s64 __u64 __int128_t __uint128_t
    __le16 __le32 __le64 __be16 __be32 __be64
    __aligned_u64 __aligned_s64 __aligned_le64 __aligned_be64
    __builtin_va_list __gnuc_va_list va_list
    FILE __FILE fpos_t off_t off64_t __off_t __off64_t _off_t _off64_t loff_t __loff_t
    time_t clock_t clockid_t timer_t suseconds_t useconds_t
    pid_t uid_t gid_t id_t mode_t dev_t ino_t nlink_t blksize_t blkcnt_t key_t
    __pid_t __uid_t __gid_t __dev_t socklen_t sa_family_t in_addr_t in_port_t
    div_t ldiv_t lldiv_t mbstate_t locale_t sig_atomic_t sigset_t __sigset_t
    siginfo_t stack_t jmp_buf sigjmp_buf rlim_t fd_mask
    u_char u_short u_int u_long ushort uint caddr_t daddr_t
    sem_t pthread_t pthread_attr_t pthread_mutex_t pthread_mutexattr_t
    pthread_cond_t pthread_condattr_t pthread_key_t pthread_once_t
    pthread_rwlock_t pthread_rwlockattr_t pthread_spinlock_t
    pthread_barrier_t pthread_barrierattr_t
    __m64 __m128 __m128d __m128i __m256 __m256d __m256i __m512 __m512d __m512i
""".split()


_MACHINES = {
    "x86_64": "x86_64",
    "amd64": "x86_64",
    "aarch64": "aarch64",
    "arm64": "aarch64",
    "riscv64": "riscv64",
    "loongarch64": "loongarch64",
}
_MACHINE = _MACHINES.get(platform.machine().lower(), "")
# Plain char is unsigned on aarch64 and riscv64, except in the Apple and
# Windows ABIs.
CHAR_UNSIGNED = _MACHINE in ("aarch64", "riscv64") and sys.platform not in ("darwin", "win32")


def _max(ctype: type, signed: bool) -> int:
    bits = 8 * ctypes.sizeof(ctype)
    return (1 << (bits - 1)) - 1 if signed else (1 << bits) - 1


# The limits macros a header may use without including limits.h or
# stdint.h, with the host's values. Suffixes give each its C type.
_LIMITS = {
    "CHAR_BIT": "8",
    "SCHAR_MIN": "(-128)",
    "SCHAR_MAX": "127",
    "UCHAR_MAX": "255",
    "CHAR_MIN": "0" if CHAR_UNSIGNED else "(-128)",
    "CHAR_MAX": "255" if CHAR_UNSIGNED else "127",
    "SHRT_MIN": "(-32767-1)",
    "SHRT_MAX": "32767",
    "USHRT_MAX": "65535",
    "INT_MIN": "(-2147483647-1)",
    "INT_MAX": "2147483647",
    "UINT_MAX": "4294967295U",
    "LONG_MIN": f"(-{_max(ctypes.c_long, True)}L-1)",
    "LONG_MAX": f"{_max(ctypes.c_long, True)}L",
    "ULONG_MAX": f"{_max(ctypes.c_ulong, False)}UL",
    "LLONG_MIN": "(-9223372036854775807LL-1)",
    "LLONG_MAX": "9223372036854775807LL",
    "ULLONG_MAX": "18446744073709551615ULL",
    "INT8_MIN": "(-128)",
    "INT8_MAX": "127",
    "UINT8_MAX": "255",
    "INT16_MIN": "(-32767-1)",
    "INT16_MAX": "32767",
    "UINT16_MAX": "65535",
    "INT32_MIN": "(-2147483647-1)",
    "INT32_MAX": "2147483647",
    "UINT32_MAX": "4294967295U",
    "INT64_MIN": "(-9223372036854775807LL-1)",
    "INT64_MAX": "9223372036854775807LL",
    "UINT64_MAX": "18446744073709551615ULL",
    "SSIZE_MAX": f"{_max(ctypes.c_ssize_t, True)}LL",
    "SIZE_MAX": f"{_max(ctypes.c_size_t, False)}ULL",
    "PTRDIFF_MIN": f"(-{_max(ctypes.c_ssize_t, True)}LL-1)",
    "PTRDIFF_MAX": f"{_max(ctypes.c_ssize_t, True)}LL",
    "INTPTR_MIN": f"(-{_max(ctypes.c_ssize_t, True)}LL-1)",
    "INTPTR_MAX": f"{_max(ctypes.c_ssize_t, True)}LL",
    "UINTPTR_MAX": f"{_max(ctypes.c_size_t, False)}ULL",
    "INTMAX_MIN": "(-9223372036854775807LL-1)",
    "INTMAX_MAX": "9223372036854775807LL",
    "UINTMAX_MAX": "18446744073709551615ULL",
}

PRELUDE = "\n".join(
    [
        f"typedef int {UNKNOWN};",
        *(f"typedef {UNKNOWN} {name};" for name in _SYSTEM_TYPES),
        "typedef _Bool bool;",
        *(f"#define {name} {value}" for name, value in _LIMITS.items()),
        "#define NULL ((void *)0)",
        "#define EXIT_SUCCESS 0",
        "#define EXIT_FAILURE 1",
        "#define SEEK_SET 0",
        "#define SEEK_CUR 1",
        "#define SEEK_END 2",
        "#define false 0",
        "#define true 1",
        "#define __bool_true_false_are_defined 1",
        "#define noreturn _Noreturn",
        "#define thread_local _Thread_local",
        "#define static_assert _Static_assert",
        "#define alignas _Alignas",
        "#define alignof _Alignof",
        "",
    ]
)

# Fixed width and other standard names map straight to ctypes, whatever
# the header's own typedef says.
STD_TYPES: dict[str, type] = {
    "int8_t": ctypes.c_int8,
    "uint8_t": ctypes.c_uint8,
    "int16_t": ctypes.c_int16,
    "uint16_t": ctypes.c_uint16,
    "int32_t": ctypes.c_int32,
    "uint32_t": ctypes.c_uint32,
    "int64_t": ctypes.c_int64,
    "uint64_t": ctypes.c_uint64,
    "int_least8_t": ctypes.c_int8,
    "uint_least8_t": ctypes.c_uint8,
    "int_least16_t": ctypes.c_int16,
    "uint_least16_t": ctypes.c_uint16,
    "int_least32_t": ctypes.c_int32,
    "uint_least32_t": ctypes.c_uint32,
    "int_least64_t": ctypes.c_int64,
    "uint_least64_t": ctypes.c_uint64,
    "size_t": ctypes.c_size_t,
    "ssize_t": ctypes.c_ssize_t,
    "ptrdiff_t": ctypes.c_ssize_t,
    "intptr_t": ctypes.c_ssize_t,
    "uintptr_t": ctypes.c_size_t,
    "intmax_t": ctypes.c_int64,
    "uintmax_t": ctypes.c_uint64,
    "wchar_t": ctypes.c_wchar,
    "char16_t": ctypes.c_uint16,
    "char32_t": ctypes.c_uint32,
    "bool": ctypes.c_bool,
    # Linux UAPI fixed width types (linux/types.h), the same on every ABI;
    # the __aligned_ ones are 8 byte aligned, as uint64_t is on 64 bit hosts.
    "__u8": ctypes.c_uint8,
    "__s8": ctypes.c_int8,
    "__u16": ctypes.c_uint16,
    "__s16": ctypes.c_int16,
    "__u32": ctypes.c_uint32,
    "__s32": ctypes.c_int32,
    "__u64": ctypes.c_uint64,
    "__s64": ctypes.c_int64,
    "__le16": ctypes.c_uint16,
    "__be16": ctypes.c_uint16,
    "__le32": ctypes.c_uint32,
    "__be32": ctypes.c_uint32,
    "__le64": ctypes.c_uint64,
    "__be64": ctypes.c_uint64,
    **(
        {
            "__aligned_u64": ctypes.c_uint64,
            "__aligned_s64": ctypes.c_int64,
            "__aligned_le64": ctypes.c_uint64,
            "__aligned_be64": ctypes.c_uint64,
        }
        if ctypes.alignment(ctypes.c_uint64) == 8
        else {}
    ),
    "time_t": ctypes.c_int64,
    "pid_t": ctypes.c_int,
}

# Attributes that do not change the layout of anything; they are dropped.
# `_Noreturn` is `noreturn` after the prelude's stdnoreturn.h macro.
HARMLESS_ATTRIBUTES = frozenset(
    """
    deprecated unavailable unused used nonnull returns_nonnull noreturn const
    pure malloc format format_arg visibility always_inline noinline gnu_inline
    artificial cold hot sentinel alloc_size access nothrow leaf may_alias
    warn_unused_result dllimport dllexport weak alias section constructor
    destructor fallthrough nodiscard warning error cleanup no_sanitize
    returns_twice no_instrument_function externally_visible assume_aligned
    nonstring optimize target flatten noclone regparm designated_init
    transparent_union selectany thread naked novtable noalias restrict
    safebuffers code_seg allocator empty_bases _Noreturn
    """.split()
)

# Attributes that change the layout and are applied: `packed`,
# `aligned(N)` and MSVC's `align(N)`.
LAYOUT_ATTRIBUTES = frozenset(["packed", "aligned", "align"])


def _keywords_common() -> list[str]:
    return [
        "__extension__",
        "__restrict",
        "__restrict__",
        "__inline",
        "__inline__",
        "__forceinline",
        "__signed__ signed",
        "__signed signed",
        "__volatile__ volatile",
        "__volatile volatile",
        "__const__ const",
        "__const const",
        "__asm__(x)",
        "__asm(x)",
        "_Static_assert(...)",
    ]


def predefined_macros() -> list[str]:
    """`NAME value` definitions for the host compiler: MSVC on Windows,
    GCC (or clang on macOS) elsewhere. Sizes come from ctypes."""
    machine = _MACHINE
    size = ctypes.sizeof
    if sys.platform == "win32":
        return [
            *_keywords_common(),
            "_MSC_VER 1940",
            "_WIN32 1",
            "_WIN64 1",
            *(["_M_X64 100", "_M_AMD64 100"] if machine == "x86_64" else []),
            *(["_M_ARM64 1"] if machine == "aarch64" else []),
            "__STDC_VERSION__ 201710L",
            "_INTEGRAL_MAX_BITS 64",
            "__cdecl",
            "__stdcall",
            "__fastcall",
            "__vectorcall",
            "__unaligned",
            "__ptr32",
            "__ptr64",
            "__w64",
            "__int8 char",
            "__int16 short",
            "__int32 int",
            "__int64 long long",
        ]
    macros = [
        *_keywords_common(),
        "__GNUC__ 14",
        "__GNUC_MINOR__ 2",
        "__GNUC_PATCHLEVEL__ 0",
        "__STDC__ 1",
        "__STDC_VERSION__ 201710L",
        "__STDC_HOSTED__ 1",
        "__CHAR_BIT__ 8",
        f"__SIZEOF_POINTER__ {size(ctypes.c_void_p)}",
        f"__SIZEOF_SHORT__ {size(ctypes.c_short)}",
        f"__SIZEOF_INT__ {size(ctypes.c_int)}",
        f"__SIZEOF_LONG__ {size(ctypes.c_long)}",
        f"__SIZEOF_LONG_LONG__ {size(ctypes.c_longlong)}",
        f"__SIZEOF_SIZE_T__ {size(ctypes.c_size_t)}",
        f"__SIZEOF_WCHAR_T__ {size(ctypes.c_wchar)}",
        f"__SIZEOF_FLOAT__ {size(ctypes.c_float)}",
        f"__SIZEOF_DOUBLE__ {size(ctypes.c_double)}",
        f"__INT_MAX__ {_max(ctypes.c_int, True)}",
        f"__LONG_MAX__ {_max(ctypes.c_long, True)}L",
        f"__LONG_LONG_MAX__ {_max(ctypes.c_longlong, True)}LL",
        f"__SIZE_MAX__ {_max(ctypes.c_size_t, False)}ULL",
        "__ORDER_LITTLE_ENDIAN__ 1234",
        "__ORDER_BIG_ENDIAN__ 4321",
        f"__BYTE_ORDER__ {1234 if sys.byteorder == 'little' else 4321}",
    ]
    if size(ctypes.c_void_p) == 8 and size(ctypes.c_long) == 8:
        macros += ["__LP64__ 1", "_LP64 1"]
    macros += {
        "x86_64": ["__x86_64__ 1", "__x86_64 1", "__amd64__ 1", "__amd64 1"],
        "aarch64": ["__aarch64__ 1"],
        "riscv64": ["__riscv 1", "__riscv_xlen 64"],
        "loongarch64": ["__loongarch64 1", "__loongarch__ 1", "__loongarch_grlen 64"],
    }.get(machine, [])
    if sys.platform == "darwin":
        macros += ["__APPLE__ 1", "__MACH__ 1", "__clang__ 1", "__clang_major__ 17"]
        if machine == "aarch64":
            macros.append("__arm64__ 1")
    else:
        if sys.platform.startswith("linux"):
            macros += ["__linux__ 1", "__linux 1", "__unix__ 1", "__unix 1", "__gnu_linux__ 1"]
            # From <asm/byteorder.h>, which is not read; the Linux UAPI
            # headers choose bit field orders with it.
            macros.append("__LITTLE_ENDIAN_BITFIELD 1" if sys.byteorder == "little" else "__BIG_ENDIAN_BITFIELD 1")
        if CHAR_UNSIGNED:
            macros.append("__CHAR_UNSIGNED__ 1")
    return macros
