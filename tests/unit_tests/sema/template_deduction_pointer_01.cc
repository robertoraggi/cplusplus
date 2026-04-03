// RUN: %cxx -verify -fcheck %s
// expected-no-diagnostics

template <class T>
T load(T const* p) {
  return *p;
}

template <class T>
T* identity_ptr(T* p) {
  return p;
}

long g_long = 42;
int g_int = 7;

// Deduction from T const*: should deduce T = long
static_assert(__is_same(decltype(load(&g_long)), long),
              "deducing T from T const* should give T = long");

// Deduction from T*: should deduce T = int
static_assert(__is_same(decltype(identity_ptr(&g_int)), int*),
              "deducing T from T* should give T = int");
