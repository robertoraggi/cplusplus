// RUN: %cxx -verify -fsyntax-only %s
// expected-no-diagnostics

struct S {
  char c;
  long long x;
};

static_assert(__alignof__(S::x) == alignof(long long));
static_assert(sizeof(S::x) == sizeof(long long));

template <typename T>
struct membuf {
  struct T2 {
    T m;
  };
  __attribute__((__aligned__(__alignof__(T2::m)))) unsigned char
      storage[sizeof(T)];
};

static_assert(alignof(membuf<double>) == alignof(double));
static_assert(alignof(membuf<char>) == 1);
