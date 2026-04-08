// RUN: %cxx -verify %s
// expected-no-diagnostics

template <class _Tp, _Tp _Xp>
struct integral_constant {
  static constexpr const _Tp value = _Xp;
};

template <class _Tp>
using lref_t = __add_lvalue_reference(_Tp);

template <class _Tp>
using rref_t = __add_rvalue_reference(_Tp);

// Direct type parameters - baseline
template <class T, class U>
struct is_nothrow_assignable_base
    : integral_constant<bool, __is_nothrow_assignable(T, U)> {};

// Type alias template instantiations as arguments
template <class _Tp>
struct is_nothrow_copy_assignable
    : integral_constant<bool, __is_nothrow_assignable(lref_t<_Tp>,
                                                      lref_t<const _Tp>)> {};

template <class _Tp>
struct is_nothrow_move_assignable
    : integral_constant<bool,
                        __is_nothrow_assignable(lref_t<_Tp>, rref_t<_Tp>)> {};

// Instantiations
is_nothrow_copy_assignable<int> t1;
is_nothrow_move_assignable<int> t2;
