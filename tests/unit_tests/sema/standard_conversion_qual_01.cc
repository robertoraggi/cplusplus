// RUN: %cxx -verify %s

// clang-format off

// expected-no-diagnostics

void test_add_const() {
  int x = 42;
  const int* p = &x;
}

void test_add_volatile() {
  int x = 42;
  volatile int* p = &x;
}

void test_add_const_volatile() {
  int x = 42;
  const volatile int* p = &x;
}

void test_multi_level_add_const() {
  int x = 0;
  int* p = &x;
  int* const* pp = &p;
}

void takes_const_ptr(const int*);

void test_func_param_qual() {
  int x = 42;
  takes_const_ptr(&x);
}

void takes_const_volatile_ptr(const volatile int*);

void test_func_param_cv_qual() {
  int x = 42;
  takes_const_volatile_ptr(&x);
}
