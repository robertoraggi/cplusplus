// RUN: %cxx -std=c++20 -verify -fsyntax-only %s
// expected-no-diagnostics

struct R {
  int value;

  template <int I>
  constexpr int get() const {
    return value + I;
  }
};

constexpr R first(int value) { return {value}; }
constexpr R second(int value) { return {value * 2}; }

template <R (*F)(int), int N>
constexpr int pointer() {
  auto r = F(N);
  return r.get<1>();
}

template <bool B, int N>
constexpr int conditional() {
  auto r = (B ? first : second)(N);
  R (*function)(int) = B ? first : second;
  auto s = function(N);
  return r.get<1>() + s.get<2>();
}

struct Callable {
  int scale;

  constexpr R operator()(int value) const { return {value * scale}; }
};

template <Callable F, int N>
constexpr int callable() {
  auto r = F(N);
  return r.get<1>();
}

struct Maker {
  constexpr R make(int value) const { return {value}; }

  template <int N>
  static constexpr R create() {
    return {N};
  }
};

template <R (Maker::*F)(int) const, int N>
constexpr int memberPointer() {
  auto r = (Maker{}.*F)(N);
  return r.get<1>();
}

template <class F>
constexpr int typeDependent(F function) {
  auto r = function(3);
  return r.template get<1>();
}

template <auto F, int N>
constexpr int placeholderParameter() {
  auto r = F(N);
  return r.template get<1>();
}

template <int N>
constexpr R create() {
  return {N};
}

template <int N>
constexpr int dependentTemplateId() {
  auto r = create<N>();
  auto s = Maker::create<N>();
  auto t = Maker{}.create<N>();
  return r.template get<1>() + s.template get<2>() + t.template get<3>();
}

static_assert(pointer<first, 3>() == 4);
static_assert(pointer<second, 3>() == 7);
static_assert(conditional<true, 3>() == 9);
static_assert(conditional<false, 3>() == 15);
static_assert(callable<Callable{2}, 3>() == 7);
static_assert(memberPointer<&Maker::make, 3>() == 4);
static_assert(typeDependent(first) == 4);
static_assert(placeholderParameter<second, 3>() == 7);
static_assert(dependentTemplateId<3>() == 15);
