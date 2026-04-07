// RUN: %cxx -toolchain macos -verify %s

unsigned _BitInt(129) a; // expected-error {{unsigned _BitInt of bit sizes greater than 128 not supported}}
unsigned _BitInt(256) b; // expected-error {{unsigned _BitInt of bit sizes greater than 128 not supported}}
_BitInt(129) c;          // expected-error {{signed _BitInt of bit sizes greater than 128 not supported}}
