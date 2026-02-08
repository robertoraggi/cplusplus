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
// CHECK-NEXT:    constructor defaulted void Base()
// CHECK-NEXT:    constructor defaulted void Base(const ::Base&)
// CHECK-NEXT:    constructor defaulted void Base(::Base&&)
// CHECK-NEXT:    function bool operator bool() const
// CHECK-NEXT:    function void f()
// CHECK-NEXT:    function defaulted void ~Base()
// CHECK-NEXT:  class Derived
// CHECK-NEXT:    base class Base
// CHECK-NEXT:    constructor defaulted void Derived()
// CHECK-NEXT:    constructor defaulted void Derived(const ::Derived&)
// CHECK-NEXT:    constructor defaulted void Derived(::Derived&&)
// CHECK-NEXT:    using bool operator bool() const
// CHECK-NEXT:    using void f()
// CHECK-NEXT:    function defaulted void ~Derived()
// CHECK-NEXT:  function int main()
// CHECK-NEXT:    block
// CHECK-NEXT:      variable static constexpr const char __func__[5]
// CHECK-NEXT:      using decltype(nullptr) nullptr_t
