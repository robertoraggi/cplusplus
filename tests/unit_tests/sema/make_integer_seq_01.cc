// clang-format off
// RUN: %cxx -verify -fcheck %s

// expected-no-diagnostics

template <class _Tp, _Tp... _Ip>
struct integer_sequence {
  typedef _Tp value_type;
  static constexpr unsigned long size() noexcept { return sizeof...(_Ip); }
};

template <unsigned long... _Ip>
using index_sequence = integer_sequence<unsigned long, _Ip...>;

template <class _Tp, _Tp _Ep>
using make_integer_sequence = __make_integer_seq<integer_sequence, _Tp, _Ep>;

template <unsigned long _Np>
using make_index_sequence = make_integer_sequence<unsigned long, _Np>;

static_assert(make_index_sequence<0>::size() == 0, "empty sequence");
static_assert(make_index_sequence<1>::size() == 1, "single element");
static_assert(make_index_sequence<5>::size() == 5, "five elements");

// Direct use
static_assert(__make_integer_seq<integer_sequence, int, 3>::size() == 3, "direct use");
