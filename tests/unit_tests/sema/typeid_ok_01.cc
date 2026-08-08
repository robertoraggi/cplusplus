// RUN: %cxx -verify -fsyntax-only %s

int value = 42;

void f() {
  (void)typeid(int);    // expected-error {{you need to include <typeinfo> before using the 'typeid' operator}}
  (void)typeid(value);  // expected-error {{you need to include <typeinfo> before using the 'typeid' operator}}
}
