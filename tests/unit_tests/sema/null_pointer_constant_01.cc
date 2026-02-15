// RUN: %cxx -verify -fcheck %s

extern "C" void abort();

// clang-format off

void test_pointer_eq_false(int* p) {
  if (p == false)
    abort();  // expected-error {{invalid operands to binary expression ('int*' and 'bool')}}
}

void test_pointer_ne_false(int* p) {
  if (p != false)
    abort();  // expected-error {{invalid operands to binary expression ('int*' and 'bool')}}
}

void test_pointer_eq_zero(int* p) {
  if (p == 0) abort();
}

void test_pointer_eq_nullptr(int* p) {
  if (p == nullptr) abort();
}

// Parenthesized null pointer constants
void test_pointer_eq_paren_zero(int* p) {
  if (p == (0)) abort();
}

void test_pointer_eq_paren_false(int* p) {
  if (p == (false))
    abort();  // expected-error {{invalid operands to binary expression ('int*' and 'bool')}}
}
