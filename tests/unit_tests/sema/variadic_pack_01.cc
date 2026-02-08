// RUN: %cxx -verify -fcheck %s

template <typename... Ts>
constexpr int count() {
  return sizeof...(Ts);
}

static_assert(count<int, float, double>() == 3, "sizeof... with 3 types");

template <typename... Ts>
struct MyTuple;

template <>
struct MyTuple<> {};

template <typename Head, typename... Tail>
struct MyTuple<Head, Tail...> {
  Head head;
  MyTuple<Tail...> tail;
};

static_assert(sizeof(MyTuple<int>) > 0, "single element tuple");
static_assert(sizeof(MyTuple<int, float>) > 0, "two element tuple");
static_assert(sizeof(MyTuple<int, float, double>) > 0, "three element tuple");

template <typename... Ts>
struct Single;

template <typename T>
struct Single<T> {
  T value;
};

static_assert(sizeof(Single<int>) == sizeof(int), "single partial spec");

template <typename... Ts>
struct TypeHolder;

template <typename T>
struct TypeHolder<T> {
  using type = T;
};

static_assert(__is_same(TypeHolder<int>::type, int), "type holder deduction");
