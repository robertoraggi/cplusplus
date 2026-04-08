// RUN: %cxx -toolchain macos -verify %s

_BitInt(-1) a;          // expected-error {{signed _BitInt of bit sizes greater than 128 not supported}}
unsigned _BitInt(-1) b; // expected-error {{unsigned _BitInt of bit sizes greater than 128 not supported}}
