// clang-format off
// RUN: %cxx -verify %s

// Template lambda IIFE with non-type parameter: static_assert fires on
// instantiation with the default argument _False=false.
// expected-note@1 {{in instantiation of function template specialization 'operator () <0>' requested here}}
inline constexpr int v1 = []<bool _False = false>() -> int {
  static_assert(_False, "should fire");  // expected-error {{"should fire"}}
  return 0;
}();

// Template lambda with non-type parameter used in body expression.
auto f1 = []<int N = 0>() -> int { return N; };

// Template lambda with type parameter used in function parameter types.
auto f2 = []<typename T>(T a, T b) -> T { return a; };

// Template lambda with multiple type parameters.
auto f3 = []<typename T, typename U>(T a, U b) { return a; };

// Template lambda with parameter pack.
auto f4 = []<typename... Ts>() -> int { return sizeof...(Ts); };

// Template lambda with non-type parameter pack.
auto f5 = []<int... Ns>() -> int { return sizeof...(Ns); };

// Constexpr variable initialized with template lambda IILE.
inline constexpr int v2 = []<int N = 42>() -> int { return N; }();
