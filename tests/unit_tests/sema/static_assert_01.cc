// RUN: %cxx -verify -fcheck %s

static_assert(true);
static_assert('a');
static_assert(1);
static_assert(0.1);
static_assert(0.1f);
static_assert("hello");

// expected-error@1 {{static assert failed}}
static_assert(false);

// expected-error@1 {{static assert failed}}
static_assert(0);

// expected-error@1 {{static assert failed}}
static_assert(0.0);

// expected-error@1 {{static assert failed}}
static_assert('\0');

// expected-error@1 {{"it will fail"}}
static_assert(false, "it will fail");

int a = 0;

// clang-format off
// expected-error@1 {{static assertion expression is not an integral constant expression}}
static_assert(a);

// clang-format on
