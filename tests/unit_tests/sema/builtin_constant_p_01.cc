// RUN: %cxx -verify -fcheck %s
// expected-no-diagnostics

static_assert(__builtin_constant_p(42));
static_assert(__builtin_constant_p(0));
static_assert(__builtin_constant_p(3.14));

static_assert(__builtin_constant_p(1 + 2));

constexpr int z = 10;
static_assert(__builtin_constant_p(z));
static_assert(__builtin_constant_p(z * 2));

static_assert(__builtin_constant_p("ciao"));

enum struct E { A = 1, B = 2 };

static_assert(__builtin_constant_p(E::A));
