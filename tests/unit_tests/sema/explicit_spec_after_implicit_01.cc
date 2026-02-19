// clang-format off
// RUN: %cxx -verify -fcheck %s
// expected-no-diagnostics

template <class _Tp = void>
struct plus {
  typedef _Tp __result_type;
};

using X = plus<void>;

template <>
struct plus<void> {
  typedef void is_transparent;
};

static_assert(sizeof(plus<void>) > 0, "explicit spec after implicit inst");
