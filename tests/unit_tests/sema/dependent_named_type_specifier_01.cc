// RUN: %cxx -verify %s
// expected-no-diagnostics

template <class _Tp, _Tp __v>
struct integral_constant {
  static constexpr _Tp value = __v;
  using value_type = _Tp;
  using type = integral_constant;
};

using true_type = integral_constant<bool, true>;
using false_type = integral_constant<bool, false>;

template <bool _Val>
using _BoolConstant = integral_constant<bool, _Val>;

template <class _Tp, class _Up>
struct is_same : _BoolConstant<__is_same(_Tp, _Up)> {};

template <class _Tp>
struct remove_cv {
  using type = __remove_cv(_Tp);
};

template <class _Tp>
using remove_cv_t = __remove_cv(_Tp);

template <class _Tp>
struct numeric_limits {
  static constexpr int digits = 0;
};

template <class _Tp>
constexpr int get_digits() {
  return numeric_limits<_Tp>::digits;
}

template <class _Tp, class _Up>
constexpr bool same_without_cv =
    is_same<remove_cv_t<_Tp>, remove_cv_t<_Up>>::value;

template <class _Tp, class _Up>
using IsSame = _BoolConstant<__is_same(_Tp, _Up)>;

static_assert(same_without_cv<const int, int>);
static_assert(!same_without_cv<int, double>);
static_assert(IsSame<int, int>::value);
static_assert(!IsSame<int, double>::value);
static_assert(get_digits<int>() == 0);
