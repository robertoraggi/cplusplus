// RUN: %cxx -verify -fcheck %s

namespace std {
inline namespace __1 {

template <class _Tp, _Tp __v>
struct integral_constant {
  static constexpr const _Tp value = __v;
};
typedef integral_constant<bool, true> true_type;
typedef integral_constant<bool, false> false_type;
template <bool _Val>
using _BoolConstant = integral_constant<bool, _Val>;
template <class _Tp, class _Up>
using _IsNotSame = _BoolConstant<!__is_same(_Tp, _Up)>;

struct __libcpp_is_referenceable_impl {
  template <class _Tp>
  static _Tp& __test(int);
  template <class _Tp>
  static false_type __test(...);
};
template <class _Tp>
struct __libcpp_is_referenceable
    : integral_constant<
          bool,
          _IsNotSame<decltype(__libcpp_is_referenceable_impl::__test<_Tp>(0)),
                     false_type>::value> {};

template <class _Up, bool = __libcpp_is_referenceable<_Up>::value>
struct __decay {
  using type = _Up;
};

template <class _Tp>
struct decay {
  using type =
      typename __decay<_Tp, __libcpp_is_referenceable<_Tp>::value>::type;
};

template <class _Tp>
using __decay_t = typename decay<_Tp>::type;

template <class _Pointer, class = void>
struct __to_address_helper;

// This should not trigger premature instantiation of decay<decltype(...)>.
// The decltype argument is dependent because _Pointer is a template parameter.
template <class _Pointer>
constexpr __decay_t<decltype(__to_address_helper<_Pointer>::__call())>
__to_address(const _Pointer& __p) noexcept {
  return {};
}

}  // namespace __1
}  // namespace std

// expected-no-diagnostics
auto main() -> int { return 0; }
