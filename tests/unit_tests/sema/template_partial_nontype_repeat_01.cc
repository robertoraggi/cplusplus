// RUN: %cxx -verify -fcheck %s
// expected-no-diagnostics

template <int N, class T>
struct pick_n {
  enum { value = 0 };
};

template <class T>
struct pick_n<0, T> {
  enum { value = 1 };
};

static_assert(pick_n<0, int>::value == 1, "non-type partial specialization");
static_assert(pick_n<7, int>::value == 0,
              "primary template for non-zero non-type argument");
