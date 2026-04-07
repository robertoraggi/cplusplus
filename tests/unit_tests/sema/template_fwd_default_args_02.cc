// RUN: %cxx -verify -fcheck %s
// expected-no-diagnostics

template <class T = int>
void func();

template <class T>
void func() {}

template <class T = void>
struct S;

template <class T>
struct S {};

template <template <class> class C = S>
struct Meta {};

void test() {
  func<>();
  func<double>();
  S<> s;
  S<int> si;
  Meta<> m;
}
