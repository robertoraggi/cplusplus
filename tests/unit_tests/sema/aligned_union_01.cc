// clang-format off
// RUN: %cxx -verify -fcheck %s

// Test: static const members with dependent initializers used as NTTP args
// in a nested template NNS should not trigger premature instantiation.

using size_t = decltype(sizeof(0));

template <size_t _I0, size_t... _In>
struct aligned_storage {
  typedef struct {
    alignas(_I0) unsigned char __data[_I0];
  } type;
};

template <size_t _I0, size_t... _In>
struct static_max;

template <size_t _I0>
struct static_max<_I0> {
  static const size_t value = _I0;
};

template <size_t _I0, size_t _I1, size_t... _In>
struct static_max<_I0, _I1, _In...> {
  static const size_t value = _I0 >= _I1 ? static_max<_I0, _In...>::value
                                         : static_max<_I1, _In...>::value;
};

namespace ns {
template <size_t _Len, class _Type0, class... _Types>
struct aligned_union {
  static const size_t alignment_value =
      static_max<alignof(_Type0), alignof(_Types)...>::value;
  static const size_t __len =
      static_max<_Len, sizeof(_Type0), sizeof(_Types)...>::value;
  // This typedef must not trigger instantiation of aligned_storage with
  // dependent __len / alignment_value.
  typedef typename aligned_storage<__len, alignment_value>::type type;
};
} // namespace ns

// Verify that static const fields in template scope used as NTTP args
// compile without errors as template definitions.
// expected-no-diagnostics
