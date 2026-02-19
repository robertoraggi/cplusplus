// clang-format off
// RUN: %cxx -verify -fcheck %s

// Test: NNS with dependent static const member used as NTTP works
// when the template is actually instantiated with concrete values.

using size_t = decltype(sizeof(0));

template <size_t _I0, size_t... _In>
struct aligned_storage {
  typedef struct {
    alignas(_I0) unsigned char __data[_I0];
  } type;
};

template <size_t N>
struct wrapper {
  static const size_t len = N;
  typedef typename aligned_storage<len>::type type;
};

// Concrete instantiation should work.
using W = wrapper<16>;
static_assert(sizeof(W::type) == 16, "wrapper<16> should have size 16");

// expected-no-diagnostics
