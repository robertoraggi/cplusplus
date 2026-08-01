// RUN: %cxx -verify -fcheck %s

// clang-format off

// expected-note@+1 {{candidate function not viable: no known conversion from 'decltype(nullptr)' to 'bool' for argument 1}}
void takes_bool(bool);

void test_int_to_bool() {
  int x = 42;
  takes_bool(x);
}

void test_zero_to_bool() {
  takes_bool(0);
}

void test_double_to_bool() {
  double d = 1.0;
  takes_bool(d);
}

void test_float_to_bool() {
  float f = 1.0f;
  takes_bool(f);
}

void test_char_to_bool() {
  char c = 'a';
  takes_bool(c);
}

enum Color { Red, Green, Blue };

void test_enum_to_bool() {
  Color c = Red;
  takes_bool(c);
}

void test_pointer_to_bool() {
  int x = 42;
  takes_bool(&x);
}

void test_null_pointer_to_bool() {
  int* p = nullptr;
  takes_bool(p);
}

void test_void_pointer_to_bool() {
  int x = 42;
  void* p = &x;
  takes_bool(p);
}

struct S { int x; };

void test_member_ptr_to_bool() {
  int S::* mp = nullptr;
  takes_bool(mp);
}

void test_member_data_ptr_to_bool() {
  int S::* mp = &S::x;
  takes_bool(mp);
}

void test_nullptr_to_bool() {
  // expected-error@+1 {{no matching function for call to 'takes_bool'}}
  takes_bool(nullptr);
}
