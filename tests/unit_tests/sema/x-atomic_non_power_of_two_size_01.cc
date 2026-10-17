// XFAIL: *
// RUN: %cxx -verify -fsyntax-only %s

// expected-no-diagnostics

struct Triple {
  char bytes[12];
};

using AtomicTriple = _Atomic(Triple);

Triple load(AtomicTriple* p) {
  return __c11_atomic_load(p, 5);
}
