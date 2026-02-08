// RUN: %cxx -verify -fcheck %s

struct String {
  String(const char* s) {}
};

void print(String s) {}

void test_string_conversion() {
  String s = "hello";
  print("world");
}

struct Num {
  int val;
  Num(int v) : val(v) {}
  operator double() { return val; }
};

void take_double(double d) {}

void test_num_conversions() {
  Num n = 42;
  double d = n;
  take_double(n);
}

struct A {};

struct B {
  B(A a) {}
};

void overloaded(B b) {}

void test_overload_with_conversion() {
  A a;
  overloaded(a);
}

struct Source {
  operator int() { return 0; }
};

void take_int_val(int x) {}
void take_long_val(long x) {}

void test_best_conversion() {
  Source s;
  take_int_val(s);
}
