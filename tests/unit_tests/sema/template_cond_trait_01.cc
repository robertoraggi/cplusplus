// RUN: %cxx -verify %s
// expected-no-diagnostics

template <class _Tp, _Tp _Xp>
struct integral_constant {
  static constexpr const _Tp value = _Xp;
};

template <class T>
struct is_copy_assignable
    : integral_constant<bool, __is_assignable(T&, const T&)> {};

template <class T>
struct is_move_assignable : integral_constant<bool, __is_assignable(T&, T&&)> {
};

template <bool B, class T, class F>
struct conditional {
  using type = T;
};

template <class T, class F>
struct conditional<false, T, F> {
  using type = F;
};

template <bool B, class T, class F>
using conditional_t = typename conditional<B, T, F>::type;

struct nat {};

template <class T>
struct pair1 {
  conditional_t<is_copy_assignable<T>::value && is_move_assignable<T>::value, T,
                void>
      member;
};

pair1<int> p1;

template <class T1, class T2>
struct pair2 {
  using first_type = T1;
  using second_type = T2;

  pair2& operator=(conditional_t<is_copy_assignable<first_type>::value &&
                                     is_copy_assignable<second_type>::value,
                                 pair2, nat> const& p) {
    return *this;
  }
};

pair2<int, int> p2;
