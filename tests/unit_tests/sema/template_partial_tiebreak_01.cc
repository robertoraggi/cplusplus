// RUN: %cxx -verify -fcheck %s

template <typename T, typename U>
struct S;

// expected-error@+2 {{partial specialization is ambiguous}}
template <typename T>
struct S<T, int> {
  enum { value = 1 };
};

template <typename U>
struct S<int, U> {
  enum { value = 2 };
};

template struct S<int, int>;
