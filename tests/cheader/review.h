/* The constructs raised in the design review; ORACLE hides what a C
   compiler rejects. */
#include <stdint.h>
#include <stddef.h>
#include "inc.h"
#include "inc.h"
#define BIG (3ULL << 30)
#define NEG (-7 % 2)
#define INV (~0U)
#define FLT 1.5
#define STR "ab" "cd"
#define DIV (-7 / 2)
#define REF (BIG + 1)
#define UNK (sizeof(int))
#if 0xffffffffffffffffULL > 0
#define BR 1
#else
#define BR 0
#endif
#if __has_builtin(__builtin_expect)
#define HB 1
#else
#define HB 0
#endif
struct al { char c; _Alignas(4) int i; };
struct al2 { char c; _Alignas(16) int i; };
struct fal { char c; int i __attribute__((aligned(4))); };
struct fal2 { char c; int i __attribute__((aligned(16))); };
struct O;
#ifndef ORACLE
struct byval { struct O o; };
#endif
struct byptr { struct O *o; };
typedef int v4si __attribute__((vector_size(16)));
struct vec { v4si v; };
typedef int __attribute__((mode(QI))) qi_t;
struct q { qi_t x; };
typedef struct { char c; } __attribute__((aligned(16))) A16;
struct P { char c; A16 a; } __attribute__((packed));
enum E { value = 1, RED = 2 };
struct WithE { char c; enum E e; };
typeof(int) bad_decl;
struct after_bad { int x; };
struct __attribute__((packed)) pk { char c; int i:4; int j:20; };
struct pk2 { char c; int i; } __attribute__((packed, deprecated("x"), aligned(2)));
static inline int f(int x) { return __builtin_expect(x, 1) ? ({ int y = x; y; }) : 0; }
int g(int a[4], int (*cb)(int));
enum __attribute__((packed)) PE { PA = 1 };
struct usepe { char c; enum PE e; };
struct fbits { unsigned : 3; unsigned x : 5; unsigned : 0; unsigned y : 2; };
struct fbits2 { unsigned : 3; unsigned x : 5; };
