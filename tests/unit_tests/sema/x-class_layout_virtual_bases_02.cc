// XFAIL: *
// RUN: %cxx -toolchain macos -std=c++26 -verify -fsyntax-only %s
// expected-no-diagnostics

struct C0 {  };
struct C1 : virtual C0 { C0 m0; };
struct C2 { C1 m0[2][2]; short m1; C1 m2[3]; };
struct C3 : virtual C0 { [[no_unique_address]] C0 m0; [[no_unique_address]] C1 m1; [[no_unique_address]] C1 m2[2]; };
struct C4 : C0, C3 { [[no_unique_address]] C3 m0; };
struct C5 : C4, C2, C0 { C3 m0; int : 0; C2 m2[2][2]; };
static_assert(sizeof(C5) == 800);
