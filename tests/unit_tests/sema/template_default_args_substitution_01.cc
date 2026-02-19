// RUN: %cxx -verify -fcheck %s
// expected-no-diagnostics

template <class T = int, int N = 4>
struct Box {
  enum { value = N };
};

static_assert(Box<>::value == 4, "both defaults");
static_assert(Box<char>::value == 4, "default non-type argument");
static_assert(Box<char, 7>::value == 7, "explicit non-type argument");
