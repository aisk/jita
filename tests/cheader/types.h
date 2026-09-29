/* Layout fixture: every struct and union here is also compiled by
   tests/test_cheader_oracle.py and compared with the C compiler. */
#ifndef TYPES_H
#define TYPES_H
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <wchar.h>

typedef long my_ssize;
typedef my_ssize my_hash;
typedef my_hash my_hash2;

struct Scalars {
    char c; signed char sc; unsigned char uc;
    short s; unsigned short us; int i; unsigned u; unsigned int ui;
    long l; unsigned long ul; long long ll; unsigned long long ull;
    float f; double d; long double ld;
    _Bool b; bool b2;
    int8_t i8; uint8_t u8; int16_t i16; uint16_t u16;
    int32_t i32; uint32_t u32; int64_t i64; uint64_t u64;
    size_t sz; ptrdiff_t pd; intptr_t ip; uintptr_t up; wchar_t wc;
    const volatile int cv;
    my_hash2 h;
};

typedef struct Vec { int64_t *data; size_t len; } Vec;
typedef struct { char tag; double value; } Kind, *PKind;

struct Pointers {
    struct Vec *v; char *s; const char *cs; void *p; void **pp;
    wchar_t *ws; int (*arr)[4]; Kind *k; unsigned char *bytes;
};

struct Arrays { int a[8]; double m[3][4]; char name[16]; Vec vs[2]; };
struct Flex { size_t n; int data[]; };

struct Node { struct Node *next; uint32_t count; };
struct A { struct B *b; int x; };
struct B { struct A a; char y; };
struct Later;
struct UsesLater { struct Later *p; };
struct Later { short s; };

struct Nested {
    int x;
    struct Inner { char c; long long l; } inner;
    union { int i; float f; };
    struct { char c2; short s2; };
    int _jita_anon0;
};

enum Color { RED, GREEN = 5, BLUE, NEG = -3, EXPR = 1 << 4, PREV = BLUE + 1 };
typedef enum { K_A = -1, K_B } KindE;
enum Wide { W_BIG = 0x80000000u };
enum Huge { H_BIG = 0x100000000LL };
enum HugeNeg { HN_NEG = -1, HN_BIG = 0x100000000LL };
struct WithEnum { char c; enum Color color; KindE k; };

struct Bits { unsigned a : 3; unsigned b : 5; int c : 1; unsigned : 4; char d; };
struct Bf2 { char a[1]; int b : 30; short c : 9; uint64_t d : 40; uint8_t e; };
struct Bf3 { uint64_t a : 40; uint32_t b : 30; uint8_t c : 7; };
struct Bf4 { uint8_t a : 3; uint32_t b : 30; };

struct __attribute__((packed)) Packed1 { char c; int i; };
struct Packed2 { char c; int i; } __attribute__((packed));
struct Aligned16 { char c; } __attribute__((aligned(16)));
typedef struct { char c; } __attribute__((aligned(16))) A16;
struct PackedA16 { char c; A16 a; } __attribute__((packed));
struct PackedAligned { char c; int i; } __attribute__((packed, deprecated("x"), aligned(2)));
struct PackedLd { char c; long double d; } __attribute__((packed));

#pragma pack(push, 1)
struct Pragma1 { char c; int i; short s; };
#pragma pack(pop)
#pragma pack(2)
struct Pragma2 { char c; int i; };
#pragma pack()
struct AfterPragma { char c; int i; };

struct FieldAlign {
    char c;
    _Alignas(4) int i;
    char d;
    _Alignas(short) short s;
    char e;
    int j __attribute__((aligned(4)));
    char f; __attribute__((aligned(2))) short t;
    char g, h __attribute__((aligned(1)));
};

typedef int (*cb_t)(void *, int);
struct Callbacks { cb_t fn; void (*raw)(void); int (*two)(int, double); char tail; };

union Num { int32_t i; double d; char bytes[12]; };

#endif
