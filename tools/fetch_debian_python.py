"""Unpack a Debian CPython for another architecture, to run under qemu-user.

    python tools/fetch_debian_python.py loong64 3.14 DEST

python-build-standalone and uv have no loongarch64 interpreter, but
Debian does. This reads the suite's package index for the Debian
architecture (`loong64`, `riscv64`, ...), resolves the dependencies of
python3.X-minimal, libpython3.X-stdlib and libpython3.X, downloads the
packages into DEST/debs and unpacks them into DEST/root. glibc and the
GCC runtime are left out: they come from the cross sysroot that qemu is
pointed at with `-L` (Debian's libc6-<arch>-cross and
libgcc-s1-<arch>-cross), as do a few packages that only matter to a
full system. With `--with-libc` they are unpacked from the same index
instead, so the interpreter and its C library always match (the cross
packages of a moving suite can lag behind the glibc its Python was built
against), and DEST/root is a sysroot of its own. Downloads are checked
against the index's SHA256 sums and kept, so a second run only unpacks.

On success the last lines of the output are shell assignments, for
`>> "$GITHUB_ENV"` or `eval`:

    PYTHON=DEST/root/usr/bin/python3.X
    PYTHON_LD_LIBRARY_PATH=DEST/root/usr/lib/<triplet>
    PYTHON_SYSROOT=DEST/root                  (with --with-libc)

and the interpreter runs as

    qemu-loongarch64 -L /usr/loongarch64-linux-gnu \\
        -E LD_LIBRARY_PATH=$PYTHON_LD_LIBRARY_PATH $PYTHON

or, with `--with-libc`, as `qemu-loongarch64 -L $PYTHON_SYSROOT $PYTHON`.

The package set follows the suite as it moves; `--mirror` can point at a
snapshot.debian.org URL to pin it. Only the standard library is needed
to unpack (.deb members compressed with gzip, xz, bzip2 or, from Python
3.14 on, zstd).
"""

import argparse
import hashlib
import io
import lzma
import re
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

MIRROR = "https://deb.debian.org/debian"

# Not needed to run the interpreter (data and tools of a full system).
SKIP_SYSTEM = {"tzdata", "media-types", "netbase", "debconf", "debconf-2.0", "dpkg", "perl-base"}

# Provided by the cross sysroot unless --with-libc: glibc and the GCC
# runtime. libatomic1 is unpacked with them; no Python package needs it,
# but code loaded into the interpreter may.
LIBC = ["libc6", "libgcc-s1", "libatomic1", "libstdc++6"]


def parse_index(text: str) -> dict[str, dict[str, str]]:
    """Package name -> fields of a Debian Packages file."""
    packages: dict[str, dict[str, str]] = {}
    for stanza in text.split("\n\n"):
        fields: dict[str, str] = {}
        key = None
        for line in stanza.splitlines():
            if line[:1] in (" ", "\t") and key is not None:
                fields[key] += "\n" + line.strip()
            elif ":" in line:
                key, value = line.split(":", 1)
                fields[key] = value.strip()
        if "Package" in fields:
            packages[fields["Package"]] = fields
    return packages


def resolve(packages: dict[str, dict[str, str]], wanted: list[str], skip: set[str]) -> list[str]:
    """The wanted packages and their Pre-Depends and Depends, recursively,
    in discovery order. Of alternatives the first one in the index (or
    provided by a package in it) is taken, and a virtual package is
    satisfied by the first package that provides it."""
    provides: dict[str, str] = {}
    for name, fields in packages.items():
        for item in fields.get("Provides", "").split(","):
            virtual = item.split("(")[0].strip()
            if virtual:
                provides.setdefault(virtual, name)
    order: list[str] = []
    seen: set[str] = set()

    def add(name: str) -> None:
        if name in skip or name in seen:
            return
        if name not in packages:
            real = provides.get(name)
            if real is None:
                raise SystemExit(f"no package or provider for {name!r}")
            name = real
            if name in skip or name in seen:
                return
        seen.add(name)
        order.append(name)
        for field in ("Pre-Depends", "Depends"):
            for dep in filter(None, (d.strip() for d in packages[name].get(field, "").split(","))):
                names = [re.split(r"[\s(:]", alt.strip(), maxsplit=1)[0] for alt in dep.split("|")]
                known = [n for n in names if n in packages or n in provides or n in skip]
                add(known[0] if known else names[0])

    for name in wanted:
        add(name)
    return order


def download(url: str, attempts: int = 3) -> bytes:
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(url, timeout=120) as response:
                return response.read()
        except OSError as e:
            if attempt == attempts:
                raise
            print(f"retrying {url} after {e}", file=sys.stderr)
            time.sleep(2 * attempt)
    raise AssertionError("unreachable")


def unpack_deb(data: bytes, root: Path) -> None:
    """Extract the data.tar.* member of a .deb (an ar archive) into root."""
    if data[:8] != b"!<arch>\n":
        raise ValueError("not an ar archive")
    pos = 8
    while pos + 60 <= len(data):
        header = data[pos : pos + 60]
        name = header[:16].decode().strip().rstrip("/")
        size = int(header[48:58].decode().strip())
        body = data[pos + 60 : pos + 60 + size]
        pos += 60 + size + (size & 1)
        if not name.startswith("data.tar"):
            continue
        if name.endswith(".zst"):
            from compression import zstd  # Python 3.14

            body = zstd.decompress(body)
        with tarfile.open(fileobj=io.BytesIO(body), mode="r:*") as tar:
            tar.extractall(root, filter="tar")
        return
    raise ValueError("no data.tar member")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("arch", help="Debian architecture, e.g. loong64")
    parser.add_argument("version", help="Python version, e.g. 3.14")
    parser.add_argument("dest", type=Path, help="directory for debs/ and root/")
    parser.add_argument("--suite", default="sid")
    parser.add_argument("--mirror", default=MIRROR)
    parser.add_argument(
        "--with-libc", action="store_true", help="also unpack glibc and the GCC runtime, making DEST/root a sysroot"
    )
    args = parser.parse_args(argv)

    index_url = f"{args.mirror}/dists/{args.suite}/main/binary-{args.arch}/Packages.xz"
    print(f"reading {index_url}", file=sys.stderr)
    packages = parse_index(lzma.decompress(download(index_url)).decode())
    ver = args.version
    wanted = [f"python{ver}-minimal", f"libpython{ver}-stdlib", f"libpython{ver}"]
    if args.with_libc:
        order = resolve(packages, wanted + LIBC, SKIP_SYSTEM)
    else:
        order = resolve(packages, wanted, SKIP_SYSTEM | set(LIBC))

    debs, root = args.dest / "debs", args.dest / "root"
    debs.mkdir(parents=True, exist_ok=True)
    root.mkdir(parents=True, exist_ok=True)
    for name in order:
        fields = packages[name]
        path = debs / Path(fields["Filename"]).name
        data = path.read_bytes() if path.exists() else b""
        if hashlib.sha256(data).hexdigest() != fields["SHA256"]:
            print(f"downloading {name} {fields['Version']}", file=sys.stderr)
            data = download(f"{args.mirror}/{fields['Filename']}")
            if hashlib.sha256(data).hexdigest() != fields["SHA256"]:
                raise SystemExit(f"{name}: SHA256 mismatch")
            path.write_bytes(data)
        unpack_deb(data, root)

    if args.with_libc:
        # Debian's merged /usr: the program interpreter is /lib64/ld-*.so
        # and the base system provides the top level links.
        for name in ("bin", "lib", "lib64", "sbin"):
            link = root / name
            if (root / "usr" / name).is_dir() and not link.exists() and not link.is_symlink():
                link.symlink_to(f"usr/{name}")
    python = root / "usr" / "bin" / f"python{ver}"
    libs = sorted(root.glob(f"usr/lib/*/libpython{ver}.so*"))
    if not python.exists() or not libs:
        raise SystemExit(f"python{ver} or libpython{ver} missing after unpacking {', '.join(order)}")
    print(f"unpacked {len(order)} packages: {' '.join(order)}", file=sys.stderr)
    print(f"PYTHON={python.resolve()}")
    print(f"PYTHON_LD_LIBRARY_PATH={libs[0].parent.resolve()}")
    if args.with_libc:
        print(f"PYTHON_SYSROOT={root.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
