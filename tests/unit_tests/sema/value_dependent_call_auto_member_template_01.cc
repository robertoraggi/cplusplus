// RUN: %cxx -std=c++20 -verify -fsyntax-only %s
// expected-no-diagnostics

struct R {
  int value;

  template <int I>
  constexpr int get() const {
    return value + I;
  }
};

constexpr R f(int value) { return {value}; }

struct Opt {
  int value;
};

constexpr R f(int value, Opt opt) { return {value + opt.value}; }

template <int N>
constexpr int eval() {
  auto r = f(N);
  return r.get<0>();
}

template <int N>
constexpr int initializers() {
  auto direct(f(N));
  auto braced{f(N)};
  const auto& reference = f(N);
  auto&& forwarding = f(N);
  decltype(auto) result = f(N);
  auto list = f(1, {N});
  auto typedList = f(1, Opt{N});
  return direct.get<1>() + braced.get<2>() + reference.get<3>() +
         forwarding.get<4>() + result.get<5>() + list.get<6>() +
         typedList.get<7>();
}

constexpr R objects[] = {{3}, {7}};

template <int N>
constexpr int references() {
  auto* pointer = &objects[N];
  auto& reference = objects[N];
  decltype(auto) result = (objects[N]);
  static_assert(__is_same(decltype(pointer), const R*));
  static_assert(__is_same(decltype(reference), const R&));
  static_assert(__is_same(decltype(result), const R&));
  return pointer->get<0>() + reference.get<1>() + result.get<2>();
}

template <class T>
constexpr int dependent(T value) {
  auto r = f(value);
  return r.template get<1>();
}

template <auto N>
constexpr int placeholderParameter() {
  auto r = f(N);
  return r.template get<1>();
}

static_assert(eval<1>() == 1);
static_assert(eval<-3>() == -3);
static_assert(initializers<2>() == 44);
static_assert(initializers<5>() == 65);
static_assert(references<0>() == 12);
static_assert(references<1>() == 24);
static_assert(dependent(8) == 9);
static_assert(placeholderParameter<4>() == 5);
