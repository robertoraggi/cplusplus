// RUN: %cxx -std=c++20 -verify -fsyntax-only %s
// expected-no-diagnostics

template <bool, class T = void>
struct Enable {};

template <class T>
struct Enable<true, T> {
  using type = T;
};

template <class T, typename Enable<(T(-1) > T(0)), int>::type = 0>
constexpr int sign(T) {
  return 1;
}

template <class T, typename Enable<(T(-1) < T(0)), int>::type = 0>
constexpr int sign(T) {
  return 2;
}

template <class F, class T>
constexpr int invoke(F function, T value) {
  return function(value);
}

template <class F>
constexpr int invokeReference(F& function) {
  return function(0);
}

constexpr auto unsignedPointer = &sign<unsigned>;
constexpr auto signedPointer = sign<int>;
constexpr auto nestedPointer = (&(sign<unsigned>));
auto& signedReference = sign<int>;
auto&& forwardedReference = sign<unsigned>;
constexpr decltype(auto) decltypePointer = &sign<int>;

static_assert(invoke(sign<unsigned>, 3u) == 1);
static_assert(invoke(sign<int>, -3) == 2);
static_assert(invoke(&sign<unsigned>, 3u) == 1);
static_assert(invokeReference(sign<int>) == 2);
static_assert(unsignedPointer(0) == 1);
static_assert(signedPointer(0) == 2);
static_assert(nestedPointer(0) == 1);
static_assert(signedReference(0) == 2);
static_assert(forwardedReference(0) == 1);
static_assert(decltypePointer(0) == 2);
static_assert(__is_same(decltype(decltypePointer), int (*const)(int)));

template <class T>
constexpr auto returnType(T) -> typename Enable<(T(-1) > T(0)), int>::type {
  return 3;
}

template <class T>
constexpr auto returnType(T) -> typename Enable<(T(-1) < T(0)), int>::type {
  return 4;
}

static_assert(invoke(returnType<unsigned>, 0u) == 3);
static_assert(invoke(returnType<int>, 0) == 4);

template <class T>
concept Sized = sizeof(T) > 0;

template <class T>
concept Large = Sized<T> && sizeof(T) >= sizeof(int);

template <class T>
constexpr int constrained(T) requires Sized<T> {
  return 5;
}

template <class T>
constexpr int constrained(T) requires Large<T> {
  return 6;
}

constexpr auto constrainedPointer = &constrained<int>;
auto& constrainedReference = constrained<int>;
static_assert(invoke(constrained<int>, 0) == 6);
static_assert(constrainedPointer(0) == 6);
static_assert(constrainedReference(0) == 6);

template <class T>
constexpr int target(T) {
  return 7;
}

template <class T>
constexpr int target(T*) {
  return 8;
}

constexpr int (*scalarTarget)(int) = &target<int>;
constexpr int (*pointerTarget)(int*) = target<int>;
static_assert(scalarTarget(0) == 7);
static_assert(pointerTarget(nullptr) == 8);

template <class T, class U>
constexpr int incomplete(U) {
  return 9;
}

template <class T>
constexpr int incomplete(T) {
  return 10;
}

template <class U>
constexpr int invokeTarget(int (*function)(U), U value) {
  return function(value);
}

constexpr int (*completed)(double) = incomplete<int>;
static_assert(completed(0) == 9);
static_assert(invokeTarget<double>(incomplete<int>, 0.0) == 9);

struct Member {
  template <class T, typename Enable<(T(-1) > T(0)), int>::type = 0>
  constexpr int sign(T) const {
    return 11;
  }

  template <class T, typename Enable<(T(-1) < T(0)), int>::type = 0>
  constexpr int sign(T) const {
    return 12;
  }
};

constexpr auto memberPointer = &Member::sign<int>;
static_assert((Member{}.*memberPointer)(0) == 12);

struct StaticMember {
  template <class T> requires Sized<T>
  static constexpr int sign(T) {
    return 13;
  }

  template <class T> requires Large<T>
  static constexpr int sign(T) {
    return 14;
  }
};

static_assert(invoke(StaticMember::sign<int>, 0) == 14);

namespace imported {
using ::sign;
}

static_assert(invoke(imported::sign<int>, 0) == 2);

template <class T>
constexpr int instantiated(T value) {
  auto function = sign<int>;
  auto dependent = sign<T>;
  return function(0) + dependent(value);
}

static_assert(instantiated(1) == 4);
static_assert(instantiated(1u) == 3);
