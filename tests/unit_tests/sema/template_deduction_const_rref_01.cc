// RUN: %cxx -verify %s
// expected-no-diagnostics

template <class T>
void fwd(T&&) {}

template <class T>
void crref(const T&&) {}

void test() {
  int i = 0;
  const int ci = 0;
  fwd(i);
  fwd(ci);
  fwd(0);
  crref(0);
}
