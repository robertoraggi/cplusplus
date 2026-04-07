// clang-format off
// RUN: %cxx -verify -fcheck %s

template <typename T>
void g(T t) {
  // expected-error@1 {{invalid member access into expression of type 'int'}}
  t.zero();
}

template <typename T>
void f(T t) {
  // expected-note@1 {{in instantiation of function template specialization 'g <int>' requested here}}
  g(t);
}

int main() {
  // expected-note@1 {{in instantiation of function template specialization 'f <int>' requested here}}
  f(0);
  return 0;
}
