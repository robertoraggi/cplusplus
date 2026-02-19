// clang-format off
// RUN: %cxx -verify -fcheck %s
// expected-no-diagnostics

template <class _Tp, _Tp __v>
struct integral_constant {
  static constexpr _Tp value = __v;
  using value_type = _Tp;
  using type = integral_constant;
};

using true_type = integral_constant<bool, true>;
using false_type = integral_constant<bool, false>;

template <bool __v>
using _BoolConstant = integral_constant<bool, __v>;

template <class _Tp, class _Up>
using _IsSame = _BoolConstant<__is_same(_Tp, _Up)>;

// Simplified decay (identity for non-reference/non-array types)
template <class _Tp>
struct __decay { using type = _Tp; };

template <class _Tp>
struct decay { using type = typename __decay<_Tp>::type; };

template <class _Tp>
using __decay_t = typename decay<_Tp>::type;

// conditional
template <bool _Bp, class _If, class _Then>
struct conditional { using type = _If; };

template <class _If, class _Then>
struct conditional<false, _If, _Then> { using type = _Then; };

template <bool _Bp, class _If, class _Then>
using __conditional_t = typename conditional<_Bp, _If, _Then>::type;

// common_type
template <class _Tp, class _Up>
struct __common_type2_imp {};

template <class ..._Tp>
struct common_type {};

template <class _Tp>
struct common_type<_Tp> : common_type<_Tp, _Tp> {};

template <class _Tp, class _Up>
struct common_type<_Tp, _Up>
    : __conditional_t<
          _IsSame<_Tp, __decay_t<_Tp>>::value &&
              _IsSame<_Up, __decay_t<_Up>>::value,
          __common_type2_imp<_Tp, _Up>,
          common_type<__decay_t<_Tp>, __decay_t<_Up>>> {};

// Instantiate — should NOT recurse infinitely.
// int and double are already decayed, so _IsSame<int, int>::value is true,
// _IsSame<double, double>::value is true, and the true branch is taken.
using CT = common_type<int, double>;
