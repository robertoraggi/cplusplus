// RUN: %cxx -verify -fcheck %s

template <int... Ns>
constexpr int nttp_count() {
  return sizeof...(Ns);
}

static_assert(nttp_count<1, 2, 3, 4, 5>() == 5, "NTTP sizeof...");

template <int... Ns>
constexpr int nttp_sum() {
  return (0 + ... + Ns);
}

static_assert(nttp_sum<100, 200, 300>() == 600, "NTTP sum");

template <typename T>
struct Wrap {
  T value;
};

template <typename... Ts>
struct WrapAll;

template <>
struct WrapAll<> {};

template <typename Head, typename... Tail>
struct WrapAll<Head, Tail...> {
  Wrap<Head> head;
  WrapAll<Tail...> tail;
};

static_assert(sizeof(WrapAll<int>) > 0, "WrapAll single");
static_assert(sizeof(WrapAll<int, float>) > 0, "WrapAll two");
static_assert(sizeof(WrapAll<int, float, double>) > 0, "WrapAll three");

// Test bool pack selection via partial specialization.
template <bool B, typename T, typename F>
struct Select;

template <typename T, typename F>
struct Select<true, T, F> {
  using type = T;
};

template <typename T, typename F>
struct Select<false, T, F> {
  using type = F;
};

static_assert(__is_same(Select<true, int, float>::type, int), "select true");
static_assert(__is_same(Select<false, int, float>::type, float),
              "select false");
