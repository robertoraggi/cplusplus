// RUN: %cxx -toolchain macos -verify %s

_BitInt(0) a;          // expected-error {{signed _BitInt must have a bit size of at least 2}}
_BitInt(1) b;          // expected-error {{signed _BitInt must have a bit size of at least 2}}
unsigned _BitInt(0) c; // expected-error {{unsigned _BitInt must have a bit size of at least 1}}
