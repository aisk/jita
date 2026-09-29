/* MSVC's __pragma operator, as CPython's headers use it under _MSC_VER. */
#define PUSH1 __pragma(pack(push, 1))
#define POP __pragma(pack(pop))
#define Py_UNUSED(name) __pragma(warning(push)) __pragma(warning(suppress: 4100)) name __pragma(warning(pop))
PUSH1
struct A { char c; int i; };
POP
struct B {
    __pragma(warning(push))
    __pragma(warning(disable: 4201))
    union { int u; float f; };
    char c;
    __pragma(warning(pop))
};
static inline int unused(int Py_UNUSED(x)) { return 0; }
__pragma(pack(push, 2)) struct C { char c; int i; }; __pragma(pack(pop))
struct D { char c; int i; };
struct E { char c; __pragma(pack(1)) int i; };
