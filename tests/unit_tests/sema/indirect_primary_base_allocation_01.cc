// RUN: %cxx -toolchain macos -std=c++26 -verify -fsyntax-only %s
// expected-no-diagnostics

struct C0 {};
struct C1 : virtual C0 {
  [[no_unique_address]] C0 m0;
};
struct C2 : C0, virtual C1 {
  int : 0;
  C0 m1[2][2];
};
struct C3 : C2, C1, C0 {
  [[no_unique_address]] C0 m0[2];
  C2 m1;
  [[no_unique_address]] C0 m2[2][2];
};
struct C4 : C3 {
  [[no_unique_address]] C0 m0[2];
};
struct FirstClaimantIsVirtual : C1, virtual C4, C2 {
  C1 m0[3];
  short m1;
  C2 m2[2];
};
static_assert(sizeof(FirstClaimantIsVirtual) == 176);
static_assert(__builtin_offsetof(FirstClaimantIsVirtual, m0) == 24);
static_assert(__builtin_offsetof(FirstClaimantIsVirtual, m2) == 80);

struct D0 {};
struct D1 : virtual D0 {};
struct D2 : virtual D1, virtual D0 {
  int : 0;
  char m1;
};
struct D4 : D1, D2, D0 {
  [[no_unique_address]] D1 m0;
};
struct MetBeforeItsClaimant : D1, virtual D4 {};
static_assert(sizeof(MetBeforeItsClaimant) == 40);

struct E0 {};
struct E1 {};
struct E2 : virtual E0, virtual E1 {};
struct E3 : E0, E2, E1 {};
struct E4 : virtual E3, E0, virtual E2 {};
struct E5 : virtual E0, virtual E3, virtual E1 {
  E4 m0;
};
struct PrimaryOfPrimary : E5, E2, E0 {
  [[no_unique_address]] E0 m0;
};
static_assert(__builtin_offsetof(PrimaryOfPrimary, m0) == 33);

struct F0 {};
struct F1 : F0 {
  F0 m0[2];
  int m1;
  char t;
};
struct F2 : F0 {
  int : 0;
  F0 m0[2];
  int m1;
  char t;
};
struct ReusesTailPadding1 : F1 {
  char c;
};
struct ReusesTailPadding2 : F2 {
  char c;
};
static_assert(__builtin_offsetof(ReusesTailPadding1, c) == 9);
static_assert(__builtin_offsetof(ReusesTailPadding2, c) == 9);
