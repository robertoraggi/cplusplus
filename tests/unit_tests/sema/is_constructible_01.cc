// RUN: %cxx -verify -fcheck %s

template <class T, T v>
struct integral_constant {
  static constexpr T value = v;
};

using true_type = integral_constant<bool, true>;
using false_type = integral_constant<bool, false>;

// __is_constructible with scalar types
static_assert(integral_constant<bool, __is_constructible(int)>::value,
              "int is default constructible");
static_assert(integral_constant<bool, __is_constructible(int, int)>::value,
              "int is constructible from int");
static_assert(integral_constant<bool, __is_constructible(double, int)>::value,
              "double is constructible from int");

// __is_constructible with pointer types
static_assert(integral_constant<bool, __is_constructible(int*)>::value,
              "pointer is default constructible");

// __is_nothrow_constructible with scalar types
static_assert(
    integral_constant<bool, __is_nothrow_constructible(int)>::value,
    "int is nothrow default constructible");
static_assert(
    integral_constant<bool, __is_nothrow_constructible(int, int)>::value,
    "int is nothrow constructible from int");

// __is_constructible with class types
struct Simple {};
static_assert(integral_constant<bool, __is_constructible(Simple)>::value,
              "Simple is default constructible");
static_assert(
    integral_constant<bool, __is_nothrow_constructible(Simple)>::value,
    "Simple is nothrow default constructible");

// __is_constructible in template context (the original bug)
template <class _Tp>
struct is_default_constructible
    : public integral_constant<bool, __is_constructible(_Tp)> {};

static_assert(is_default_constructible<int>::value,
              "int is default constructible via trait");
static_assert(is_default_constructible<Simple>::value,
              "Simple is default constructible via trait");

// __is_nothrow_constructible in template context
template <class _Tp, class... _Args>
struct is_nothrow_constructible
    : public integral_constant<bool,
                               __is_nothrow_constructible(_Tp, _Args...)> {};

static_assert(is_nothrow_constructible<int>::value,
              "int is nothrow constructible via trait");

// void is not constructible
static_assert(!integral_constant<bool, __is_constructible(void)>::value,
              "void is not constructible");
