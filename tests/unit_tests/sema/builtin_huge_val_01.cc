// RUN: %cxx -verify -fcheck %s
// expected-no-diagnostics
static_assert(__builtin_huge_val() > 1e308);
static_assert(__builtin_huge_valf() > 1e38f);
void test() {
  double d = __builtin_huge_val();
  float f = __builtin_huge_valf();
  long double ld = __builtin_huge_vall();
  (void)d;
  (void)f;
  (void)ld;
}
