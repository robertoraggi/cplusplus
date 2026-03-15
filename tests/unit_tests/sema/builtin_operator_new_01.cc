// RUN: %cxx -verify -fcheck %s
// expected-no-diagnostics
void test() {
  void* p = __builtin_operator_new(16);
  (void)p;
}
