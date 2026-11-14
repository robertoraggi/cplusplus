// XFAIL: *
// RUN: %cxx -toolchain macos -std=c++26 -verify -fsyntax-only %s
// expected-no-diagnostics

struct C0 {  };
struct C1 {  };
struct C2 : virtual C0, C1 { [[no_unique_address]] C0 m0[2]; [[no_unique_address]] C1 m1[3]; int : 0; };
struct C3 : virtual C2 {  };
struct C4 { [[no_unique_address]] C0 m0; char m1; };
struct C5 : C0, C4 { [[no_unique_address]] C0 m0; [[no_unique_address]] C0 m1[2][2]; };
struct C6 : C0, C3, C5 { [[no_unique_address]] C4 m0[3]; };
struct C7 : C6, C4 { int : 0; };
struct C8 { [[no_unique_address]] C6 m0[2][2]; };
struct C9 : C8 {  };
struct C10 { [[no_unique_address]] C1 m0; short m1; [[no_unique_address]] C7 m2[2][2]; };
struct C11 : virtual C1, C3, virtual C8 { char m0; [[no_unique_address]] C4 m1[2][2]; [[no_unique_address]] C6 m2; };
struct C12 {  };
struct C13 : C11, virtual C1, C8 {  };
static_assert(sizeof(C13) == 456);
