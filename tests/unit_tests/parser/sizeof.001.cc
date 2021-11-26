// RUN: %cxx -verify -fsyntax-only %s -o -

static_assert(sizeof(bool) == 1);
static_assert(alignof(bool) == 1);

static_assert(sizeof(char) == 1);
static_assert(sizeof(char signed) == 1);
static_assert(sizeof(char unsigned) == 1);
static_assert(sizeof(char const) == 1);

static_assert(sizeof(short) == 2);
static_assert(sizeof(short signed) == 2);
static_assert(sizeof(short unsigned) == 2);
static_assert(sizeof(short const) == 2);

static_assert(sizeof(int) == 4);
static_assert(sizeof(int signed) == 4);
static_assert(sizeof(int unsigned) == 4);
static_assert(sizeof(int const) == 4);

static_assert(sizeof(long) == 8);
static_assert(sizeof(long signed) == 8);
static_assert(sizeof(long unsigned) == 8);
static_assert(sizeof(long const) == 8);

static_assert(sizeof(long long) == 8);
static_assert(sizeof(long long signed) == 8);
static_assert(sizeof(long long unsigned) == 8);
static_assert(sizeof(long long const) == 8);

static_assert(sizeof(char8_t) == 1);
static_assert(sizeof(char16_t) == 2);
static_assert(sizeof(char32_t) == 4);
static_assert(sizeof(wchar_t) == 4);

static_assert(sizeof(float) == 4);
static_assert(sizeof(double) == 8);
static_assert(sizeof(long double) == 16);
static_assert(sizeof(__float128) == 16);

static_assert(sizeof(void*) == 8);

static_assert(sizeof(char[4]) == 4);
static_assert(sizeof(int[4]) == 16);

struct S {
  char c;
  int i;
};

static_assert(sizeof(S) == 8);

struct K {};

static_assert(sizeof(K) == 1);

S s;

static_assert(sizeof(s) == 8);
static_assert(sizeof(s) == sizeof(S));

enum E { none };

static_assert(sizeof(E) == 4);
