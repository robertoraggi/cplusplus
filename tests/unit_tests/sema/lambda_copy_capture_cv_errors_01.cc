// RUN: %cxx -std=c++20 -verify -fsyntax-only %s

void f(int n, int* pointer, int (&array)[2], const int fixed) {
  auto copy = [=] { n = 1; }; // expected-error {{cannot assign to a non-modifiable lvalue of type 'const int'}}
  auto named = [n] { n = 1; }; // expected-error {{cannot assign to a non-modifiable lvalue of type 'const int'}}
  auto init = [b = n] { b = 1; }; // expected-error {{cannot assign to a non-modifiable lvalue of type 'const int'}}
  auto ptr = [pointer] { pointer = nullptr; }; // expected-error {{cannot assign to a non-modifiable lvalue of type 'int* const'}}
  auto elements = [array] { array[0] = 1; }; // expected-error {{cannot assign to a non-modifiable lvalue of type 'const int'}}
  auto qualified = [fixed]() mutable { fixed = 1; }; // expected-error {{cannot assign to a non-modifiable lvalue of type 'const int'}}
  auto nested = [n] { [&n] { n = 1; }(); }; // expected-error {{cannot assign to a non-modifiable lvalue of type 'const int'}}
  auto mutableCopy = [n]() mutable { n = 1; };
  auto ref = [&n] { n = 1; };
  auto refInit = [&b = n] { b = 1; };
  auto pointee = [pointer] { *pointer = 1; };
}
