// RUN: %cxx -fcheck -dump-symbols %s | %filecheck %s

int printf(const char*, ...);

namespace std {
using ::printf;
}

struct Base {
  operator bool() const;

  void f();
};

struct Derived : Base {
  using Base::operator bool;
  using Base::f;
};

namespace std {
using nullptr_t = decltype(nullptr);
}  // namespace std

auto main() -> int {
  using std::nullptr_t;
  return 0;
}

// clang-format off
//      CHECK:namespace
// CHECK-NEXT:  function int printf(const char*...)
// CHECK-NEXT:  namespace std
// CHECK-NEXT:    using int printf(const char*...)
// CHECK-NEXT:    typealias decltype(nullptr) nullptr_t
// CHECK-NEXT:  class Base
// CHECK-NEXT:    function  operator bool() const
// CHECK-NEXT:    function void f()
// CHECK-NEXT:  class Derived
// CHECK-NEXT:    using  operator bool() const
// CHECK-NEXT:    using void f()
// CHECK-NEXT:  function int main()
// CHECK-NEXT:    block
// CHECK-NEXT:      using decltype(nullptr) nullptr_t

