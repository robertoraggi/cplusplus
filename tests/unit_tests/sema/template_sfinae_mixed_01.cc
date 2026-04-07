// RUN: %cxx -verify %s
// expected-no-diagnostics

template <bool B, class T = void>
struct enable_if {};

template <class T>
struct enable_if<true, T> {
  typedef T type;
};

template <class T>
typename enable_if<sizeof(T) == 4, int>::type foo() {
  return 10;
}

template <class T>
typename enable_if<sizeof(T) == 8, int>::type foo() {
  return 20;
}

int foo(int x) { return x; }

int main() {
  int a = foo<int>();
  int b = foo<long long>();
  int c = foo(42);
  return (a == 10 && b == 20 && c == 42) ? 0 : 1;
}
