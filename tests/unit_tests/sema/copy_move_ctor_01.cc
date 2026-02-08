// RUN: %cxx -verify -fcheck %s

struct Simple {
  int x;
  double y;
};

void test_copy() {
  Simple a;
  a.x = 1;
  a.y = 2.0;
  Simple b = a;
  Simple c(a);
}

struct WithCtor {
  int val;
  WithCtor(int v) : val(v) {}
};

void test_copy_with_ctor() {
  WithCtor a(42);
  WithCtor b = a;
  WithCtor c(a);
}

void take_by_value(Simple s) {}

void test_pass_by_value() {
  Simple s;
  s.x = 10;
  take_by_value(s);
}
