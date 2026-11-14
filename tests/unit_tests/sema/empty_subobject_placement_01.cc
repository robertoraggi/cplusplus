// RUN: %cxx -verify -fcheck %s
// expected-no-diagnostics

struct E {};
struct F {};

struct W {
  char c;
  E e;
};

struct S {
  W w;
  [[no_unique_address]] E arr[2];
};
static_assert(sizeof(S) == 4);
static_assert(__builtin_offsetof(S, arr) == 2);

struct T {
  W w;
  [[no_unique_address]] E one;
};
static_assert(sizeof(T) == 2);
static_assert(__builtin_offsetof(T, one) == 0);

struct U : E {
  char c;
  [[no_unique_address]] E arr[3];
};
static_assert(sizeof(U) == 4);
static_assert(__builtin_offsetof(U, arr) == 1);

struct V {
  [[no_unique_address]] E a;
  [[no_unique_address]] E b[2][2];
  char c;
};
static_assert(sizeof(V) == 6);
static_assert(__builtin_offsetof(V, b) == 1);
static_assert(__builtin_offsetof(V, c) == 5);

struct Chain0 : E {};
struct Chain1 : Chain0 {};
struct Chain2 : Chain1 {};

struct Mixed : Chain2, F {
  [[no_unique_address]] E e;
  [[no_unique_address]] F f;
  [[no_unique_address]] Chain1 chain;
};
static_assert(sizeof(Mixed) == 3);
static_assert(__builtin_offsetof(Mixed, e) == 1);
static_assert(__builtin_offsetof(Mixed, f) == 1);
static_assert(__builtin_offsetof(Mixed, chain) == 2);

struct Nested0 {
  [[no_unique_address]] E e;
};
struct Nested1 {
  [[no_unique_address]] Nested0 n;
  [[no_unique_address]] E e;
};
struct Nested2 {
  [[no_unique_address]] Nested1 n;
  [[no_unique_address]] E e;
};
static_assert(sizeof(Nested2) == 3);
static_assert(__builtin_offsetof(Nested2, e) == 2);

struct Elements {
  Nested1 items[3];
  [[no_unique_address]] E e;
};
static_assert(sizeof(Elements) == 7);
static_assert(__builtin_offsetof(Elements, e) == 6);

struct VirtualBase : virtual E {};
struct VirtualDerived : VirtualBase, virtual F {
  [[no_unique_address]] E e;
  [[no_unique_address]] VirtualBase member;
};
static_assert(sizeof(VirtualDerived) == 3 * sizeof(void*));
static_assert(__builtin_offsetof(VirtualDerived, e) == 0);
static_assert(__builtin_offsetof(VirtualDerived, member) == sizeof(void*));

struct Repeated : E, F {
  [[no_unique_address]] Mixed first;
  [[no_unique_address]] Mixed second;
};
static_assert(sizeof(Repeated) == 7);
static_assert(__builtin_offsetof(Repeated, first) == 1);
static_assert(__builtin_offsetof(Repeated, second) == 4);
