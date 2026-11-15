// RUN: %cxx -std=c++20 -verify -fsyntax-only %s
// expected-no-diagnostics

namespace std {
class type_info;
}

template <class T>
struct always_false {
  static constexpr bool value = false;
};

template <class T>
T&& declval() noexcept {
  static_assert(always_false<T>::value, "declval in an evaluated context");
}

struct Plain {};

template <class T>
int bySizeof(int n = sizeof(declval<T>())) {
  return n;
}

template <class T>
int byAlignof(int n = __alignof__(declval<T>())) {
  return n;
}

template <class T>
bool byNoexcept(bool b = noexcept(declval<T>())) {
  return b;
}

template <class T>
int byDecltype(int n = decltype(declval<T>() + 0){}) {
  return n;
}

template <class T>
const std::type_info& byTypeid(
    const std::type_info& info = typeid(declval<Plain&>())) {
  return info;
}

template <class T>
bool byRequires(bool b = requires { declval<T>(); }) {
  return b;
}

template <class I, class P = bool (*)(decltype(*declval<I>()))>
bool byLambdaParameter(I it,
                       P p = [](decltype(*declval<I>())) { return true; }) {
  return p(*it);
}

template <class T>
struct Aggregate {
  int n = sizeof(declval<T>());
};

template <class T>
struct WithConstructor {
  int n = sizeof(declval<T>());
  WithConstructor() {}
};

int use(int* it) {
  Aggregate<float> aggregate{};
  WithConstructor<double> withConstructor;
  (void)byTypeid<int>();
  return bySizeof<char>() + byAlignof<signed char>() + byNoexcept<short>() +
         byDecltype<long>() + byRequires<unsigned>() + byLambdaParameter(it) +
         aggregate.n + withConstructor.n;
}
