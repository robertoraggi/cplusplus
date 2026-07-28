// RUN: %cxx -verify -fcheck %s

namespace ns {
int y;
}

struct S {
  static int m;
};

void undeclared_qualifier() {
  ns2::x << 1;  // expected-error {{nested name specifier must be a class or namespace}}
}

void missing_member() {
  ns::x << 1;    // expected-error {{no member named 'x' in namespace 'ns'}}
  S::nope << 1;  // expected-error {{no member named 'nope' in 'S'}}
}

void missing_operator_function_id() {
  S::operator+;   // expected-error {{no member named 'operator +' in 'S'}}
  ns::operator+;  // expected-error {{no member named 'operator +' in namespace 'ns'}}
  operator+(1, 2);  // expected-error {{use of undeclared identifier 'operator +'}}
}

namespace ns {
void g(int);
}

struct T {
  static void h(int);
};

void missing_callee() {
  ns::nope(1);  // expected-error {{no member named 'nope' in namespace 'ns'}}
  T::nope(1);   // expected-error {{no member named 'nope' in 'T'}}
  __builtin_nosuchthing(1);  // expected-error {{unknown builtin function '__builtin_nosuchthing'}}
}

void resolved() {
  ns::y << 1;
  S::m << 1;
  ns::g(1);
  T::h(1);
}
