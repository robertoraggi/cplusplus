// RUN: %cxx -verify -fcheck %s
// expected-no-diagnostics
void test(int n) {
  void* p = __builtin_alloca(n);
  (void)p;
}
