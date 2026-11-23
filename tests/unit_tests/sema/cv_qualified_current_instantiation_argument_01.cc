// RUN: %cxx -verify -fsyntax-only %s

template <class T>
struct Vector {};

template <class T>
struct Ptr {
  using Element = T;
};

template <template <class> class Template, class T>
using Apply = Template<T>;

template <class T>
struct X {
  using Self = X<T>;
  using ConstSelf = const Self;
  using VolatileSelf = volatile Self;
  using MutablePtr = Ptr<Self>;
  using ConstPtr = Ptr<ConstSelf>;
  using VolatilePtr = Ptr<VolatileSelf>;
  using ConstVolatilePtr = Ptr<const VolatileSelf>;

  static_assert(__is_same(MutablePtr, Ptr<X>));
  static_assert(__is_same(ConstPtr, Ptr<const X<T>>));
  static_assert(__is_same(Apply<X, int>, X<int>));
  static_assert(!__is_same(MutablePtr, ConstPtr));
  static_assert(!__is_same(ConstPtr, VolatilePtr));
  static_assert(!__is_same(ConstPtr, ConstVolatilePtr));

  constexpr int f(const Vector<MutablePtr>&);
  constexpr int f(const Vector<ConstPtr>&);
  constexpr int f(const Vector<VolatilePtr>&) { return 3; }
  constexpr int f(const Vector<ConstVolatilePtr>&) { return 4; }
};

template <class T>
constexpr int X<T>::f(const Vector<MutablePtr>&) {
  return 1;
}

template <class T>
constexpr int X<T>::f(const Vector<ConstPtr>&) {
  return 2;
}

static_assert(__is_same(X<int>::MutablePtr::Element, X<int>));
static_assert(__is_same(X<int>::ConstPtr::Element, const X<int>));
static_assert(__is_same(X<int>::VolatilePtr::Element, volatile X<int>));
static_assert(
    __is_same(X<int>::ConstVolatilePtr::Element, const volatile X<int>));

static_assert(X<int>{}.f(Vector<Ptr<X<int>>>{}) == 1);
static_assert(X<int>{}.f(Vector<Ptr<const X<int>>>{}) == 2);
static_assert(X<int>{}.f(Vector<Ptr<volatile X<int>>>{}) == 3);
static_assert(X<int>{}.f(Vector<Ptr<const volatile X<int>>>{}) == 4);
