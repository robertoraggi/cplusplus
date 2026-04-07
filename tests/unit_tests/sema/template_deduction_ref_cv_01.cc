// RUN: %cxx -verify %s
// expected-no-diagnostics

template <class T>
T& ref_identity(T& t) { return t; }

void test() {
  int i = 0;
  const int ci = 0;

  int& r1 = ref_identity(i);
  const int& r2 = ref_identity(ci);
}
