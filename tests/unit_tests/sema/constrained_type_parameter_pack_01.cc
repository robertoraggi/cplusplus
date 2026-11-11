// RUN: %cxx -std=c++26 -verify -fsyntax-only %s
// expected-no-diagnostics

template <class T, class U>
concept Same = __is_same(T, U);

template <Same<float>... Ts>
constexpr int f(Ts...) {
  return 1;
}

template <Same<double>... Ts>
constexpr int f(Ts...) {
  return 2;
}

template <class... Ts>
concept CanCall = requires(Ts... values) { f(values...); };

static_assert(f(1.0f) == 1);
static_assert(f(1.0, 2.0) == 2);
static_assert(f(1.0f, 2.0f, 3.0f) == 1);
static_assert(f<float>(1.0) == 1);
static_assert(f<float>(1.0f, 2.0f) == 1);
static_assert(f<double>(1.0, 2.0) == 2);
static_assert(!CanCall<int>);
static_assert(!CanCall<float, double>);
static_assert(!CanCall<double, float>);
static_assert(!CanCall<float, float, double>);
static_assert(!CanCall<>);

constexpr int (*floatPointer)(float, float) = f<float, float>;
constexpr auto doublePointer = &f<double, double>;
static_assert(floatPointer(1, 2) == 1);
static_assert(doublePointer(1, 2) == 2);

template <Same<float>... Ts>
constexpr int fallback(Ts...) {
  return 1;
}

template <class... Ts>
constexpr int fallback(Ts...) {
  return 0;
}

static_assert(fallback() == 1);
static_assert(fallback(1.0f, 2.0f) == 1);
static_assert(fallback(1.0f, 2.0) == 0);
static_assert(fallback(1.0, 2.0f) == 0);

template <class T>
concept Never = false;

template <Never... Ts>
constexpr int vacuous(Ts...) {
  return sizeof...(Ts);
}

template <class... Ts>
concept CanVacuous = requires(Ts... values) { vacuous(values...); };

static_assert(vacuous() == 0);
static_assert(!CanVacuous<int>);

template <class T>
concept HasType = requires { typename T::type; };

struct WithType {
  using type = int;
};

struct WithoutType {};

template <HasType... Ts>
constexpr int hasTypes(Ts...) {
  return 1;
}

template <class... Ts>
constexpr int hasTypes(Ts...) {
  return 0;
}

static_assert(hasTypes() == 1);
static_assert(hasTypes(WithType{}, WithType{}) == 1);
static_assert(hasTypes(WithType{}, WithoutType{}) == 0);
static_assert(hasTypes(WithoutType{}, WithType{}) == 0);

template <class T, class U, int N>
concept SameSize = Same<T, U> && sizeof(T) == N;

template <SameSize<float, sizeof(float)>... Ts>
constexpr int explicitArguments(Ts...) {
  return sizeof...(Ts);
}

static_assert(explicitArguments() == 0);
static_assert(explicitArguments(1.0f, 2.0f) == 2);

template <class T, class... Us>
concept AnyOf = (Same<T, Us> || ...);

template <class... Us>
struct Nested {
  template <AnyOf<Us...>... Ts>
  static constexpr int accepts(Ts...) {
    return 1;
  }

  template <class... Ts>
  static constexpr int accepts(Ts...) {
    return 0;
  }
};

static_assert(Nested<int, float>::accepts(1, 2.0f, 3) == 1);
static_assert(Nested<int, float>::accepts(1, 2.0) == 0);
static_assert(Nested<int, float>::accepts(1.0, 2.0f) == 0);
static_assert(Nested<>::accepts() == 1);
static_assert(Nested<>::accepts(1) == 0);

template <Same<float>... Ts>
struct FloatTypes {};

template <class... Ts>
concept CanForm = requires { typename FloatTypes<Ts...>; };

static_assert(CanForm<>);
static_assert(CanForm<float, float>);
static_assert(!CanForm<float, double>);

template <class... Ts>
struct Types {};

template <Same<float>... Ts>
using FloatAlias = Types<Ts...>;

template <class... Ts>
concept CanAlias = requires { typename FloatAlias<Ts...>; };

static_assert(CanAlias<>);
static_assert(CanAlias<float, float>);
static_assert(!CanAlias<double, float>);

template <class T>
concept IsFloat = Same<T, float>;

constexpr int abbreviated(IsFloat auto...) { return 1; }
constexpr int abbreviated(auto...) { return 0; }

static_assert(abbreviated() == 1);
static_assert(abbreviated(1.0f, 2.0f) == 1);
static_assert(abbreviated(1.0f, 2.0) == 0);
static_assert(abbreviated(1.0, 2.0f) == 0);

constexpr auto lambda = [](IsFloat auto... values) { return sizeof...(values); };

template <class... Ts>
concept CanLambda = requires(Ts... values) { lambda(values...); };

static_assert(lambda() == 0);
static_assert(lambda(1.0f, 2.0f) == 2);
static_assert(!CanLambda<float, double>);
static_assert(!CanLambda<double, float>);

template <class... Ts>
struct Choice {
  static constexpr int value = 0;
};

template <Same<float>... Ts>
struct Choice<Ts...> {
  static constexpr int value = 1;
};

template <Same<double>... Ts>
struct Choice<Ts...> {
  static constexpr int value = 2;
};

static_assert(Choice<float, float>::value == 1);
static_assert(Choice<double, double>::value == 2);
static_assert(Choice<float, double>::value == 0);
static_assert(Choice<int>::value == 0);

template <Same<float> T, Same<double>... Ts>
constexpr int multiple(T, Ts...) {
  return 1;
}

template <class T, class... Ts>
constexpr int multiple(T, Ts...) {
  return 0;
}

static_assert(multiple(1.0f) == 1);
static_assert(multiple(1.0f, 2.0, 3.0) == 1);
static_assert(multiple(1.0, 2.0) == 0);
static_assert(multiple(1.0f, 2.0f) == 0);

template <Same<float>... Ts, class U>
constexpr int trailing(U, Ts...) {
  return sizeof...(Ts);
}

template <class... Ts>
concept CanTrailing = requires(Ts... values) { trailing(1, values...); };

static_assert(trailing(1) == 0);
static_assert(trailing(1, 2.0f, 3.0f) == 2);
static_assert(!CanTrailing<float, double>);
static_assert(CanCall<const float>);
static_assert(CanCall<float&>);
static_assert(!CanForm<const float>);
static_assert(!CanForm<float&>);
