// RUN: %cxx -verify %s

// clang-format off

// expected-no-diagnostics

void takes_int(int);

void test_char_promotes() {
  char c = 'a';
  takes_int(c);
}

void test_signed_char_promotes() {
  signed char sc = 1;
  takes_int(sc);
}

void test_unsigned_char_promotes() {
  unsigned char uc = 1;
  takes_int(uc);
}

void test_short_promotes() {
  short s = 1;
  takes_int(s);
}

void test_unsigned_short_promotes() {
  unsigned short us = 1;
  takes_int(us);
}

void test_char8_t_promotes() {
  char8_t c = u8'a';
  takes_int(c);
}

void test_char16_t_promotes() {
  char16_t c = u'a';
  takes_int(c);
}

void test_char32_t_promotes() {
  char32_t c = U'a';
  takes_int(c);
}

void test_wchar_t_promotes() {
  wchar_t c = L'a';
  takes_int(c);
}

void test_bool_promotes() {
  bool b = true;
  takes_int(b);
}

enum Color { Red, Green, Blue };

void test_unscoped_enum_promotes() {
  Color c = Red;
  takes_int(c);
}

enum SmallEnum : unsigned char { X, Y, Z };

void test_enum_with_underlying_promotes() {
  SmallEnum e = X;
  takes_int(e);
}
