// RUN: %cxx -verify -fcheck -dump-symbols %s

template <typename T>
constexpr bool is_integral_v = __is_integral(T);

static_assert(is_integral_v<int>);
static_assert(is_integral_v<char>);
static_assert(is_integral_v<void*> == false);

// clang-format off
//      CHECK:namespace
// CHECK-NEXT:  template variable constexpr const bool is_integral_v
// CHECK-NEXT:    parameter typename<0, 0> T
// CHECK-NEXT:    [specializations]
// CHECK-NEXT:      variable constexpr bool is_integral_v<int>
// CHECK-NEXT:      variable constexpr bool is_integral_v<char>
// CHECK-NEXT:      variable constexpr bool is_integral_v<void*>
