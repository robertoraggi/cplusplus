// RUN: %cxx -std=c++26 -verify -fsyntax-only %s
// expected-no-diagnostics

struct Z {
  int z = 4;
  constexpr virtual int id() const { return z; }
};
struct A : virtual Z {
  int a = 1;
  constexpr A() = default;
  constexpr A(int v) : a(v) {}
  constexpr int id() const override { return a + z; }
};
struct P { int p = 8; };
struct B : P, virtual A {
  int b = 2;
  constexpr B() : A(10), b(id() + A::id() + a) {}
};
struct C : virtual A {
  int c = 3;
  constexpr C() : A(20), c(z * 100) { z = 6; }
  constexpr int id() const override { return c; }
};
struct D : B, C {
  constexpr D() : A(30) {}
  constexpr int id() const override { return a + b + c + z + p; }
};

constexpr D d{};
static_assert(d.a == 30);
static_assert(d.z == 6);
static_assert(d.p == 8);
static_assert(d.b == 34 + 34 + 30);
static_assert(d.c == 400);
static_assert(static_cast<const Z&>(d).id() == 30 + 98 + 400 + 6 + 8);

constexpr B b{};
static_assert(b.b == 14 + 14 + 10);

struct Counter {
  int* count;
  constexpr ~Counter() { ++*count; }
};
struct Owner : virtual Counter {
  constexpr Owner(int* c) : Counter{c} {}
};
struct Most : Owner {
  constexpr Most(int* c) : Counter{c}, Owner(c) {}
};
constexpr int destroyed() {
  int count = 0;
  { Most m(&count); }
  { Owner o(&count); }
  return count;
}
static_assert(destroyed() == 2);

struct Heap {
  int* value = new int(5);
  constexpr ~Heap() { delete value; }
};
struct HeapOwner : virtual Heap {};
struct HeapMost : HeapOwner {};
constexpr int heapValue() {
  HeapMost m;
  return *m.value;
}
static_assert(heapValue() == 5);
