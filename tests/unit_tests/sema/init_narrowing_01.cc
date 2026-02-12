// RUN: %cxx -verify -fcheck %s

// clang-format off

struct S {
  int i;
  float f;
  short s;
};

void test_float_to_int() {
  // expected-warning@1 {{narrowing conversion from 'double' to 'int' in braced-init-list}}
  int a = {1.5};
}

void test_double_to_float() {
  // expected-warning@1 {{narrowing conversion from 'double' to 'float' in braced-init-list}}
  float b = {1.5};
}

void test_int_to_short() {
  int x = 42;
  // expected-warning@1 {{narrowing conversion from 'int' to 'short' in braced-init-list}}
  short c = {x};
}

void test_int_to_float_struct() {
  int x = 42;
  // expected-warning@1 {{narrowing conversion from 'int' to 'float' in braced-init-list}}
  struct S s1 = {0, x, 0};
}

void test_array_narrowing() {
  double d = 1.0;
  // expected-warning@1 {{narrowing conversion from 'double' to 'float' in braced-init-list}}
  float arr[1] = {d};
}

void test_no_narrowing() {
  int a = {42};
  double c = {1.0};
  long d = {42};
}

union U {
  int i;
  float f;
};

void test_union_narrowing() {
  double d = 1.0;
  // expected-warning@1 {{narrowing conversion from 'double' to 'int' in braced-init-list}}
  union U u = {d};
}

void test_designated_narrowing() {
  double d = 1.0;
  // expected-warning@1 {{narrowing conversion from 'double' to 'int' in braced-init-list}}
  struct S s = {.i = d};
}
