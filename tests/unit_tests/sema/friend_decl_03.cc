// clang-format off
// RUN: %cxx -verify -fsyntax-only %s

class K;

struct X {
  friend class K;  // K already declared, not newly introduced
};

K* k_ptr;  // OK: K was declared before the friend declaration
