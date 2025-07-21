// RUN: %cxx -verify -fcheck %s

static_assert('0');
static_assert(('0'));
static_assert('0' + 1);
static_assert(1 && 2);
static_assert(0 || "ciao");
static_assert(1 + 0);
static_assert(1 - 0 + 1 == 2);

enum E { A = 123 };
static_assert(A == 122 + 1);

static_assert(1 & 1);
static_assert(0b1 | 0b10 == 0b11);
static_assert(1 ^ 1 == 0);
static_assert(1 + 1 != 2 - 1);

static_assert(0x8000'0000 * 2 == 0);