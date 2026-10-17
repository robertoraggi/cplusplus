// RUN: %cxx -verify -fsyntax-only %s

// expected-no-diagnostics

struct Base {
  int b;
  constexpr Base(int v) noexcept;
  constexpr Base() noexcept;
};
constexpr Base::Base(int v) noexcept : b(v * 2) {}
constexpr Base::Base() noexcept : Base(21) {}

struct Member {
  long long m;
  constexpr explicit Member(long long v) noexcept;
};
constexpr Member::Member(long long v) noexcept : m{v + 1} {}

struct Derived : Base {
  Member member;
  int plain = 5;
  int arr[3];
  constexpr Derived(int x, long long y) noexcept;
  constexpr Derived(const Derived&) noexcept;
  constexpr Derived() noexcept;
};
constexpr Derived::Derived(int x, long long y) noexcept : Base(x), member(y), arr{x, x + 1, x + 2} {}
constexpr Derived::Derived(const Derived& o) noexcept : Base(o.b), member(o.member.m), plain(o.plain + 100), arr{1, 2, 3} {}
constexpr Derived::Derived() noexcept : Derived(7, 8) {}

constexpr Derived d(3, 4);
static_assert(d.b == 6 && d.member.m == 5 && d.plain == 5 && d.arr[2] == 5);
constexpr Derived e;
static_assert(e.b == 14 && e.member.m == 9 && e.arr[0] == 7);
constexpr Derived f(d);
static_assert(f.b == 12 && f.plain == 105 && f.arr[1] == 2);
constexpr Base g;
static_assert(g.b == 42);

template <class T>
struct Box {
  T value;
  T twice;
  constexpr Box(T v) noexcept;
};
template <class T>
constexpr Box<T>::Box(T v) noexcept : value(v), twice(v + v) {}
constexpr Box<int> bi(21);
static_assert(bi.twice == 42);
constexpr Box<long long> bl(1LL << 40);
static_assert(bl.twice == (1LL << 41));

struct Ns { struct Inner { int a; constexpr Inner(int v) noexcept; }; };
constexpr Ns::Inner::Inner(int v) noexcept : a(v + 1) {}
constexpr Ns::Inner in(1);
static_assert(in.a == 2);

struct Ref {
  const int& r;
  constexpr Ref(const int& v) noexcept;
};
constexpr int global = 9;
constexpr Ref::Ref(const int& v) noexcept : r(v) {}
constexpr Ref ref(global);
static_assert(ref.r == 9);

struct Cond {
  int v;
  constexpr Cond(bool c) noexcept;
};
constexpr Cond::Cond(bool c) noexcept : v(c ? 1 : 2) {}
static_assert(Cond(true).v == 1 && Cond(false).v == 2);


struct A { int x = 3; int y; constexpr A() noexcept; };
constexpr A::A() noexcept = default;
struct B { int x = 4; constexpr B() noexcept; constexpr B(const B&) noexcept; };
constexpr B::B() noexcept = default;
constexpr B::B(const B&) noexcept = default;
constexpr B b0;
constexpr B b1(b0);
static_assert(b1.x == 4);

struct T {
  long long v;
  int tag = 11;
  template <class I> constexpr T(I i) noexcept;
};
template <class I> constexpr T::T(I i) noexcept : v(i + 1) {}
constexpr T t1(5), t2(5LL), t3((short)5);
static_assert(t1.v == 6 && t2.v == 6 && t3.v == 6 && t1.tag == 11);

struct U {
  union { int i; float f; };
  constexpr U(int v) noexcept;
};
constexpr U::U(int v) noexcept : i(v) {}
constexpr U u(8);
static_assert(u.i == 8);

struct C {
  const int c;
  int& r;
  constexpr C(int& ref, int v) noexcept;
};
constexpr int target = 1;
constexpr C::C(int& ref, int v) noexcept : c(v), r(ref) {}

struct Wrap {
  int v;
  constexpr Wrap(int x) noexcept;
  constexpr Wrap(Wrap&& o) noexcept;
};
constexpr Wrap::Wrap(int x) noexcept try : v(x) {} catch (...) {}
constexpr Wrap::Wrap(Wrap&& o) noexcept : v(o.v + 1) {}
constexpr Wrap w1(1);
static_assert(w1.v == 1);

namespace N {
struct In { int a; int b = 2; constexpr In(int v) noexcept; };
}
constexpr N::In::In(int v) noexcept : a(v) {}
static_assert(N::In(7).a == 7 && N::In(7).b == 2);

struct Local {
  int a;
  constexpr Local(int v) noexcept;
};
constexpr Local::Local(int v) noexcept : a(v) {}
constexpr int use() { Local l(9); Local m(l); return l.a + m.a; }
static_assert(use() == 18);


int main() { return d.b + e.b + f.b - 6 - 14 - 12; }
