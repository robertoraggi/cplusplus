// RUN: %cxx -verify -fcheck %s
// expected-no-diagnostics

template <class T>
void f(T&&) {}

int i = 0;
const int ci = 0;

void test() {
  f(i);
  f(0);
  f(ci);
}
