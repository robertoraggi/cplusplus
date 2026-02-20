// clang-format off
// RUN: %cxx -verify -fcheck %s

template <typename T>
struct S {
  using X = typename T::value_type; // expected-error {{type 'int' cannot be used prior to '::' because it has no members}}
};

auto main() -> int {
  S<int>::X x; // expected-note {{in instantiation of template class 'S<int>' requested here}}
  return 0;
}
