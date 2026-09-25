"""Read and write a ctypes.Structure from generated code with `typed`.

`typed(reg, Type)` views the memory at `reg` as a ctypes Structure: each
field becomes a sized memory operand at the offset ctypes computes, nested
structures and arrays become further views, and `p.items[rcx]` indexes an
array with a register.

The function below walks a linked list of `Node`s, sums every node's
`items[0..count)` into its `total` field and returns the grand total.

Run with `uv run python examples/typed_struct.py`.
"""

import ctypes

from jita import Assembler
from jita.tools.listing import listing
from jita.x64 import *  # noqa: F403


class Node(ctypes.Structure):
    pass


Node._fields_ = [
    ("next", ctypes.POINTER(Node)),
    ("count", ctypes.c_uint32),
    ("flags", ctypes.c_uint16),
    ("total", ctypes.c_int64),
    ("items", ctypes.c_int32 * 8),
]


def build() -> Assembler:
    a = Assembler()
    node = typed(rdi, Node)
    with a:
        # int64_t sum_list(Node *node)
        xor(eax, eax)  # grand total
        label("node")
        test(rdi, rdi)
        jz("done")
        xor(edx, edx)  # this node's total
        mov(r8d, node.count)
        xor(ecx, ecx)
        label("item")
        cmp(ecx, r8d)
        jae("stored")
        movsxd(r9, node.items[rcx])  # dword[rdi + rcx*4 + items offset]
        add(rdx, r9)
        inc(ecx)
        jmp("item")
        label("stored")
        mov(node.total, rdx)
        or_(node.flags, 1)  # word operand, size from the field
        add(rax, rdx)
        mov(rdi, node.next)  # pointer fields are plain qwords
        jmp("node")
        label("done")
        ret()
    return a


def main() -> None:
    nodes = [Node(count=n) for n in (3, 8, 0)]
    for i, n in enumerate(nodes):
        n.items[:] = [10 * i + k for k in range(8)]
        if i + 1 < len(nodes):
            n.next = ctypes.pointer(nodes[i + 1])
    a = build()
    with a.load() as mod:
        sum_list = mod.function(ctypes.c_int64, ctypes.POINTER(Node))
        total = sum_list(ctypes.pointer(nodes[0]))
    print(next(line for line in listing(a).splitlines() if "movsxd" in line))
    print(f"totals = {[n.total for n in nodes]}, flags = {[n.flags for n in nodes]}")
    print(f"sum = {total}")


if __name__ == "__main__":
    main()
