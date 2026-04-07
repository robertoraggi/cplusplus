// RUN: %cxx -verify %s

// clang-format off

// expected-no-diagnostics

void test_nullptr_to_int_ptr() {
  int* p = nullptr;
}

void test_nullptr_to_void_ptr() {
  void* p = nullptr;
}

void test_zero_to_pointer() {
  int* p = 0;
}

void test_nullptr_to_function_ptr() {
  void (*fp)(int) = nullptr;
}

void test_int_ptr_to_void() {
  int x = 42;
  void* p = &x;
}

void test_double_ptr_to_void() {
  double d = 1.0;
  void* p = &d;
}

struct S { int x; };

void test_class_ptr_to_void() {
  S s;
  void* p = &s;
}

void test_const_ptr_to_const_void() {
  const int x = 42;
  const void* p = &x;
}

struct Base {};
struct Derived : Base {};

void test_derived_to_base() {
  Derived d;
  Base* bp = &d;
}

struct Derived2 : Derived {};

void test_multi_level_inheritance() {
  Derived2 d;
  Base* bp = &d;
  Derived* dp = &d;
}
