// RUN: %cxx -verify -fcheck %s

template <int A, int B>
struct pick_n {
  enum { value = 0 };
};

template <class T, class U>
struct pick {
  enum { value = 0 };
};

// expected-error@+2 {{partial specialization is ambiguous}}
template <class T>
struct pick<T, int> {
  enum { value = 1 };
};

template <class U>
struct pick<int, U> {
  enum { value = 2 };
};

static_assert(pick_n<1, 0>::value == 0);
static_assert(pick<char, int>::value == 1);
static_assert(pick<int, char>::value == 2);
template struct pick<int, int>;
