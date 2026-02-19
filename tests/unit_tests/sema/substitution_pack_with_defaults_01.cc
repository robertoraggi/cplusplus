// RUN: %cxx -verify -fcheck %s
// expected-no-diagnostics

// Class template with all-defaulted parameters.
template <class T = int, int N = 4>
struct Box {
  enum { value = N };
};

// Both defaults.
static_assert(Box<>::value == 4, "both defaults");

// Override first, default second.
static_assert(Box<char>::value == 4, "second default");

// Override both.
static_assert(Box<char, 7>::value == 7, "no defaults");

// Multiple defaulted type + non-type params.
template <class A = int, class B = double, int M = 10>
struct Multi {
  enum { val = M };
};

static_assert(Multi<>::val == 10, "all defaults");
static_assert(Multi<char>::val == 10, "two defaults");
static_assert(Multi<char, float>::val == 10, "one default");
static_assert(Multi<char, float, 3>::val == 3, "no defaults");
