// RUN: %cxx -verify -ftemplates -fcheck %s

template <typename Key, typename Value>
struct X {
  Key key;
  Value value;

  void check() { static_assert(sizeof(*this)); }
};

auto main() -> int {
  using U = X<char, int>;
  U u;

  static_assert(sizeof(U) == 8);
  static_assert(sizeof(u) == 8);

  static_assert(__builtin_offsetof(U, key) == 0);
  static_assert(__builtin_offsetof(U, value) == 4);

  static_assert(sizeof(X<char, int>::key) == 1);
  static_assert(sizeof(X<char, int>::value) == 4);

  static_assert(sizeof(U::key) == 1);
  static_assert(sizeof(U::value) == 4);

  return 0;
}