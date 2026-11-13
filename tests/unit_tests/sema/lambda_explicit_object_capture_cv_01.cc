// RUN: %cxx -std=c++23 -verify -fsyntax-only %s
// expected-no-diagnostics

struct A {
  int* get();
  const int* get() const;
};

void test(A a) {
  auto nonConst = [a](this auto& self) { return a.get(); };
  auto constant = [a](this const auto& self) { return a.get(); };
  static_assert(__is_same(decltype(nonConst()), int*));
  static_assert(__is_same(decltype(constant()), const int*));
  auto forwarded = [a](this auto&& self) { return a.get(); };
  static_assert(__is_same(decltype(forwarded()), int*));
  const auto copy = forwarded;
  static_assert(__is_same(decltype(copy()), const int*));
}

struct Result {
  template <int I>
  int get() const { return I; }
};
Result target();

int functionReference() {
  Result (&reference)() = target;
  auto copy = [reference](this auto&& self) {
    auto result = reference();
    return result.template get<0>();
  };
  return copy();
}

template <class T, class U>
constexpr bool same = false;
template <class T>
constexpr bool same<T, T> = true;

void dependentObject(A a) {
  auto local = [a](this auto& self) {
    using R = decltype(a.get());
    return R{};
  };
  static_assert(same<decltype(local()), int*>);
  const auto constLocal = local;
  static_assert(same<decltype(constLocal()), const int*>);
  auto generic = [a](this const auto& self, auto) { return a.get(); };
  static_assert(same<decltype(generic(0)), const int*>);
}
