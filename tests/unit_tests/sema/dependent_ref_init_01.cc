// RUN: %cxx -verify -fcheck %s
// expected-no-diagnostics

template <class T>
void fwd_bind(T&& x) {
  T& ref = x;
  (void)ref;
}

template <class T>
void lref_bind(T& x) {
  T& ref = x;
  (void)ref;
}

void test() {
  int i = 0;
  fwd_bind(i);
  lref_bind(i);
}
