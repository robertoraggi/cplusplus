// RUN: %cxx -verify -fcheck %s

template <typename T, typename U>
struct S;

template <typename T>
struct S<T, int> {
  enum { value = 1 };
};

template <typename U>
struct S<int, U> {
  enum { value = 2 };
};

static_assert(S<int, int>::value == 1);
