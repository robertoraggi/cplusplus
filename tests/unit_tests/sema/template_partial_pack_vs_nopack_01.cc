// RUN: %cxx -verify -fcheck %s

template <class... Ts>
struct TypeList {};

template <class T>
struct S;

template <class T>
struct S<TypeList<T>> {
  enum { value = 1 };
};

template <class... Ts>
struct S<TypeList<Ts...>> {
  enum { value = 2 };
};

static_assert(S<TypeList<int>>::value == 1);
static_assert(S<TypeList<int, int>>::value == 2);
