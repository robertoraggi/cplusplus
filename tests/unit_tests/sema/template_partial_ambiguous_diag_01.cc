// RUN: %cxx -verify -fcheck %s

template <class T, class U>
struct S;

// expected-error@+2 {{partial specialization is ambiguous}}
template <class T>
struct S<T, int> {
  enum { value = 1 };
};

template <class U>
struct S<int, U> {
  enum { value = 2 };
};

template struct S<int, int>;
