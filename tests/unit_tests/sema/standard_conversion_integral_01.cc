// RUN: %cxx -verify %s

// clang-format off

// expected-no-diagnostics

void test_int_to_unsigned_long() {
  int x = 42;
  unsigned long ul = x;
}

void test_unsigned_to_long_long() {
  unsigned u = 42u;
  long long ll = u;
}

void test_long_to_short() {
  long l = 42L;
  short s = l;
}

void test_int_to_char() {
  int x = 65;
  char c = x;
}

void test_long_long_to_int() {
  long long ll = 42LL;
  int x = ll;
}

void test_int_to_unsigned() {
  int x = 42;
  unsigned u = x;
}

void test_unsigned_to_int() {
  unsigned u = 42u;
  int x = u;
}

void test_bool_to_int_via_integral() {
  bool b = true;
  long long ll = b;
}

void test_char_to_long_long() {
  char c = 'a';
  long long ll = c;
}

void test_unsigned_char_to_short() {
  unsigned char uc = 1;
  short s = uc;
}
