// RUN: %cxx -verify -fcheck %s
// expected-no-diagnostics

template <class T, int N>
struct array {
  T& operator[](int);
};

template <class T>
T&& move(T&&);

template <class Up>
struct pair_from_array {
  Up first;

  pair_from_array(array<Up, 2>&& p) : first(move(p)[0]) {}
};
