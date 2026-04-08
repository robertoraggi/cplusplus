// RUN: %cxx -verify %s

// clang-format off

void test_lvalue_to_rvalue() {
  int x = 42;
  int y = x;
}

void test_array_to_pointer() {
  int arr[5] = {};
  int* p = arr;
}

void f(int);
void test_function_to_pointer() {
  void (*fp)(int) = f;
}

void takes_int(int);
void test_integral_promotion() {
  char c = 'a';
  takes_int(c);
  short s = 1;
  takes_int(s);
  bool b = true;
  takes_int(b);
}

void takes_double(double);
void test_floating_point_promotion() {
  float f = 1.0f;
  takes_double(f);
}

void takes_long(long);
void test_integral_conversion() {
  int x = 42;
  takes_long(x);
}

void takes_long_double(long double);
void test_floating_point_conversion() {
  double d = 1.0;
  takes_long_double(d);
}

void test_floating_integral_conversion() {
  int x = 42;
  double d = x;
  int y = d;
}

void test_pointer_conversion() {
  int* p = 0;
  int* q = nullptr;
  void* v = p;
}

struct Base {};
struct Derived : Base {};

void test_derived_to_base_pointer() {
  Derived d;
  Base* bp = &d;
}

void takes_bool(bool);
void test_boolean_conversion() {
  int x = 42;
  takes_bool(x);
  double d = 1.0;
  takes_bool(d);
  int* p = nullptr;
  takes_bool(p);
}

void takes_const_int_ptr(const int*);
void test_qualification_conversion() {
  int x = 42;
  takes_const_int_ptr(&x);
}

struct FromInt {
  FromInt(int x) {}
};

void takes_from_int(FromInt f) {}

void test_user_conversion_ctor() {
  takes_from_int(42);
}

struct ToInt {
  operator int() const { return 0; }
};

void test_user_conversion_func() {
  ToInt t;
  int x = t;
}

void test_usual_arithmetic() {
  int x = 1;
  double d = 2.0;
  auto r = x + d;
}

// expected-no-diagnostics
