// RUN: %cxx -verify %s

// clang-format off

// expected-no-diagnostics

void test_float_to_long_double() {
  float f = 1.0f;
  long double ld = f;
}

void test_long_double_to_double() {
  long double ld = 1.0L;
  double d = ld;
}

void test_long_double_to_float() {
  long double ld = 1.0L;
  float f = ld;
}

void test_unsigned_to_float() {
  unsigned u = 42u;
  float f = u;
}

void test_long_long_to_double() {
  long long ll = 100LL;
  double d = ll;
}

void test_short_to_float() {
  short s = 1;
  float f = s;
}

void test_float_to_int() {
  float f = 3.14f;
  int x = f;
}

void test_double_to_long() {
  double d = 2.71;
  long l = d;
}

void test_float_to_unsigned() {
  float f = 42.0f;
  unsigned u = f;
}

void test_double_to_long_long() {
  double d = 100.0;
  long long ll = d;
}
