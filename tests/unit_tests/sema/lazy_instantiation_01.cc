// RUN: %cxx -fcheck -dump-symbols %s | %filecheck %s

template <typename T>
struct S {
  T value;
  auto get() -> T;
};

S<int>* ptr = nullptr;

// clang-format off
//      CHECK:namespace
// CHECK-NEXT:  template class S<type-param<0, 0>>
// CHECK-NEXT:    parameter typename<0, 0> T
// CHECK-NEXT:    injected class name S
// CHECK-NEXT:    field type-param<0, 0> value
// CHECK-NEXT:    function type-param<0, 0> get()
// CHECK-NEXT:    [specializations]
// CHECK-NEXT:      class S<int>
// CHECK-NEXT:  variable ::S<int>* ptr
