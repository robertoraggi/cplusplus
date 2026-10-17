// RUN: %cxx -verify -fsyntax-only %s

// expected-no-diagnostics

template <class T> concept Small = sizeof(T) <= 4;
template <class T> concept Big = sizeof(T) > 4;

template <class T> int f(T) requires Small<T>;
template <class T> int f(T) requires Big<T>;
template <class T> int f(T) requires Small<T> { return 1; }
template <class T> int f(T) requires Big<T> { return 2; }

template <class T> auto g(T v) -> T requires (sizeof(T) > 0);
template <class T> auto g(T v) -> T requires (sizeof(T) > 0) { return v; }

template <class T, class U> int h(T, U) requires (sizeof(T) == sizeof(U));
template <class T, class U> int h(T, U) requires (sizeof(T) == sizeof(U)) { return 3; }

struct S {
  template <class T> int m(T) requires Small<T>;
  template <class T> int m(T) requires Big<T>;
  template <class T> friend int fr(T, S) requires Small<T>;
};
template <class T> int S::m(T) requires Small<T> { return 4; }
template <class T> int S::m(T) requires Big<T> { return 5; }
template <class T> int fr(T, S) requires Small<T>;
template <class T> int fr(T, S) requires Small<T> { return 6; }

int main() {
  S s;
  if (f(1) != 1 || f(1.0) != 2) return 1;
  if (g(3) != 3) return 2;
  if (h(1, 2) != 3) return 3;
  if (s.m('c') != 4 || s.m(1.0) != 5) return 4;
  if (fr('c', s) != 6) return 5;
  return 0;
}
