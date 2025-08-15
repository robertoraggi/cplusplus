// RUN: %cxx -verify -fcheck -dump-symbols %s | %filecheck %s

template <typename T>
struct A {
  T a;
  A<T>* next;

  auto get_a() -> T;
  void set_a(const T& a);

  using type = T;
  using reference = T&;
};

A<int> a1;
A<void*> a2;

// clang-format off
//      CHECK:namespace
// CHECK-NEXT:  template class A<type-param<0, 0>>
// CHECK-NEXT:    parameter typename<0, 0> T
// CHECK-NEXT:    field type-param<0, 0> a
// CHECK-NEXT:    field ::A* next
// CHECK-NEXT:    function type-param<0, 0> get_a()
// CHECK-NEXT:    function void set_a(const type-param<0, 0>&)
// CHECK-NEXT:    typealias type-param<0, 0> type
// CHECK-NEXT:    typealias type-param<0, 0>& reference
// CHECK-NEXT:    [specializations]
// CHECK-NEXT:      class A<int>
// CHECK-NEXT:        field int a
// CHECK-NEXT:        field ::A<int>* next
// CHECK-NEXT:        function int get_a()
// CHECK-NEXT:        function void set_a(const int&)
// CHECK-NEXT:        typealias int type
// CHECK-NEXT:        typealias int& reference
// CHECK-NEXT:      class A<void*>
// CHECK-NEXT:        field void* a
// CHECK-NEXT:        field ::A<void*>* next
// CHECK-NEXT:        function void* get_a()
// CHECK-NEXT:        function void set_a(void* const&)
// CHECK-NEXT:        typealias void* type
// CHECK-NEXT:        typealias void*& reference
// CHECK-NEXT:  variable ::A<int> a1
// CHECK-NEXT:  variable ::A<void*> a2
