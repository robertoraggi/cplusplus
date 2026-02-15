// RUN: %cxx -fcheck -dump-symbols %s | %filecheck %s --match-full-lines

struct X {
  int i;

  X();
  X(int);
  ~X();

  operator int();
  operator const int&();
};

X::X() {}

X::X(int) {}

X::~X() {}

X::operator const int&() { return i; }

X::operator int() { return i; }

// clang-format off
//      CHECK:namespace
// CHECK-NEXT:  class X
// CHECK-NEXT:    constructor void X()
// CHECK-NEXT:      [redeclarations]
// CHECK-NEXT:        constructor void X()
// CHECK-NEXT:          block
// CHECK-NEXT:            variable static constexpr const char __func__[2]
// CHECK-NEXT:    constructor void X(int)
// CHECK-NEXT:      [redeclarations]
// CHECK-NEXT:        constructor void X(int)
// CHECK-NEXT:          parameters
// CHECK-NEXT:            parameter int
// CHECK-NEXT:            block
// CHECK-NEXT:              variable static constexpr const char __func__[2]
// CHECK-NEXT:    constructor defaulted void X(const ::X&)
// CHECK-NEXT:    constructor defaulted void X(::X&&)
// CHECK-NEXT:    field int i
// CHECK-NEXT:    function void ~X()
// CHECK-NEXT:      [redeclarations]
// CHECK-NEXT:        function void ~X()
// CHECK-NEXT:          block
// CHECK-NEXT:            variable static constexpr const char __func__[3]
// CHECK-NEXT:    function int operator int()
// CHECK-NEXT:      [redeclarations]
// CHECK-NEXT:        function int operator int()
// CHECK-NEXT:          block
// CHECK-NEXT:            variable static constexpr const char __func__[13]
// CHECK-NEXT:    function const int& operator const int&()
// CHECK-NEXT:      [redeclarations]
// CHECK-NEXT:        function const int& operator const int&()
// CHECK-NEXT:          block
// CHECK-NEXT:            variable static constexpr const char __func__[20]
