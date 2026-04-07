// RUN: %cxx -verify %s
// expected-no-diagnostics
void test() {
  double d = __builtin_nans("");
  float f = __builtin_nansf("");
  long double ld = __builtin_nansl("");
  (void)d;
  (void)f;
  (void)ld;
}
