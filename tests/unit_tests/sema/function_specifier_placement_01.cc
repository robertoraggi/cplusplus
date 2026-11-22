// RUN: %cxx -verify -fsyntax-only %s

explicit void f();  // expected-error {{'explicit' can only appear on a constructor or conversion function declared in its class definition}}
virtual void g();  // expected-error {{'virtual' can only appear on non-static member functions}}

struct S {
  explicit S(int);
  explicit operator bool();
  explicit void h();  // expected-error {{'explicit' can only appear on a constructor or conversion function declared in its class definition}}
  static virtual void s();  // expected-error {{'virtual' can only appear on non-static member functions}}
  friend virtual void k();  // expected-error {{'virtual' can only appear on non-static member functions}}
  virtual void v();
};

explicit S::S(int) {}  // expected-error {{'explicit' can only appear on a constructor or conversion function declared in its class definition}}
virtual void S::v() {}  // expected-error {{'virtual' can only appear in the initial declaration of a member function}}

template <class T>
struct X {
  explicit X(int);
  explicit operator bool();
};

template <>
struct X<int> {
  explicit X(int);
  explicit void h();  // expected-error {{'explicit' can only appear on a constructor or conversion function declared in its class definition}}
};

X<char> xc{1};

template <class T>
struct D {
  D(T);
};

explicit D(int) -> D<int>;

struct A {
  struct B;
};

struct A::B {
  explicit B(int);
};
