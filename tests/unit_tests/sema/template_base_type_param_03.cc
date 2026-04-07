// RUN: %cxx -verify %s
// expected-no-diagnostics

struct A {
  int a;
};
struct B {
  int b;
};

template <typename Base1, typename Base2>
struct Multi : public Base1, public Base2 {
  Multi() : Base1(), Base2() {}
};

Multi<A, B> m1;

template <typename T, typename Base>
struct Explicit : public Base {
  Explicit() : Base() {}
};

Explicit<int, A> e1;

template <typename T>
struct Inner {
  Inner() = default;
  explicit Inner(T) {}
  T val;
};

template <typename T>
struct Outer : public Inner<T> {
  Outer() = default;
  explicit Outer(T v) : Inner<T>(v) {}
};

Outer<int> o1;
