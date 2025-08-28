// RUN: %cxx -verify -fcheck -dump-symbols %s

template <typename T>
constexpr bool is_integral_v = __is_integral(T);

static_assert(is_integral_v<int>);
static_assert(is_integral_v<char>);
static_assert(is_integral_v<void*> == false);
