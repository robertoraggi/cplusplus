// RUN: %cxx -verify -fcheck %s

struct B {
  int x;
};

struct D : B {
  int y;
};

struct Unrelated {};

void test_invalid_static_cast() {
  Unrelated u;
  B b;

  // expected-error@1 {{invalid static_cast of '::Unrelated*' to '::D*'}}
  D* pd = static_cast<D*>(&u);

  const B cb;
  // expected-error@1 {{invalid static_cast of 'const ::B' to '::D&'}}
  D& rd = static_cast<D&>(cb);
}

void test_invalid_const_cast() {
  int i = 42;
  int* pi = &i;

  // expected-error@1 {{invalid const_cast of 'int*' to 'float*'}}
  float* pf = const_cast<float*>(pi);

  // expected-error@1 {{invalid const_cast of 'int' to 'int'}}
  int x = const_cast<int>(i);
}

void test_invalid_reinterpret_cast() {
  int i = 42;
  const int* cpi = &i;

  // expected-error@1 {{invalid reinterpret_cast of 'const int*' to 'int*'}}
  int* pi = reinterpret_cast<int*>(cpi);
}
