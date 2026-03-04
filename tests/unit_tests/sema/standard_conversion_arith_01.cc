// RUN: %cxx -verify -fcheck %s

// clang-format off

// expected-no-diagnostics


void test_long_double_dominates() {
  long double ld = 1.0L;
  int x = 1;
  auto r1 = ld + x;
  float f = 2.0f;
  auto r2 = ld + f;
  double d = 3.0;
  auto r3 = ld + d;
}

void test_double_dominates() {
  double d = 1.0;
  int x = 1;
  auto r1 = d + x;
  float f = 2.0f;
  auto r2 = d + f;
}

void test_float_dominates_int() {
  float f = 1.0f;
  int x = 1;
  auto r = f + x;
}

void test_signed_unsigned_same_rank() {
  int x = 1;
  unsigned u = 2u;
  auto r = x + u;
}

void test_long_vs_int() {
  long l = 1L;
  int x = 2;
  auto r = l + x;
}

void test_unsigned_long_vs_int() {
  unsigned long ul = 1UL;
  int x = 2;
  auto r = ul + x;
}

void test_char_plus_char() {
  char a = 'a';
  char b = 'b';
  auto r = a + b;
}

void test_short_plus_short() {
  short a = 1;
  short b = 2;
  auto r = a + b;
}
