// RUN: %cxx -verify %s
// expected-no-diagnostics

template <typename _Tp>
struct __cxx_atomic_base_impl {
  __cxx_atomic_base_impl() noexcept = default;

  constexpr explicit __cxx_atomic_base_impl(_Tp __value) noexcept
      : __a_value(__value) {}

  _Tp __a_value;
};

template <typename _Tp, typename _Base = __cxx_atomic_base_impl<_Tp> >
struct __cxx_atomic_impl : public _Base {
  __cxx_atomic_impl() noexcept = default;
  constexpr explicit __cxx_atomic_impl(_Tp __value) noexcept : _Base(__value) {}
};
