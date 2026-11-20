// RUN: %cxx -std=c++26 -verify -fsyntax-only %s
// expected-no-diagnostics

struct A {
  int a = 1;
  constexpr A() = default;
  constexpr A(int v) : a(v) {}
  constexpr virtual int id() const { return a; }
};
struct B : virtual A {
  int b = 2;
  constexpr B() : A(10) {}
  constexpr int id() const override { return a * 100 + b; }
};
struct C : virtual A {
  int c = 3;
  constexpr C() : A(20) { c += a; }
};
struct D : B, C {
  constexpr D() : A(30) {}
  constexpr int id() const override { return a + b + c; }
};

constexpr int writeThroughB() {
  D d;
  static_cast<B&>(d).a = 5;
  return static_cast<C&>(d).a;
}
static_assert(writeThroughB() == 5);

constexpr D d{};
static_assert(d.a == 30);
static_assert(static_cast<const B&>(d).a == 30);
static_assert(static_cast<const C&>(d).a == 30);
static_assert(d.c == 33);
static_assert(static_cast<const A&>(d).id() == 30 + 2 + 33);
static_assert(static_cast<const B&>(d).id() == 30 + 2 + 33);

constexpr B b{};
static_assert(b.a == 10 && b.b == 2);
static_assert(static_cast<const A&>(b).id() == 1002);

constexpr C c{};
static_assert(c.a == 20 && c.c == 23);

constexpr D copy = d;
static_assert(copy.a == 30 && copy.c == 33);
static_assert(static_cast<const A&>(copy).id() == 65);

constexpr B slice = d;
static_assert(slice.a == 30 && slice.b == 2);
static_assert(static_cast<const A&>(slice).id() == 3002);

constexpr int heap() {
  auto p = new D;
  int r = static_cast<A*>(p)->id();
  delete p;
  return r;
}
static_assert(heap() == 65);

constexpr int array() {
  D items[2];
  static_cast<C&>(items[1]).a = 7;
  return items[0].a * 10 + items[1].a;
}
static_assert(array() == 307);

constexpr const A* pointer = &d;
static_assert(pointer->a == 30);
static_assert(pointer->id() == 65);
