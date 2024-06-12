// RUN: %cxx -verify -fcheck -ftemplates -dump-symbols %s | %filecheck %s

namespace std {
using size_t = decltype(sizeof(0));

template <typename T>
struct allocator {
  using value_type = T;

  template <typename U>
  struct rebind {
    using other = allocator<U>;
  };

  auto allocate(size_t n) -> T*;
  void deallocate(T* p, size_t n);
};
}  // namespace std

std::allocator<int>::rebind<char> alloc8;
std::allocator<int>::rebind<short> alloc16;

// clang-format off
//      CHECK:namespace
// CHECK-NEXT:  namespace std
// CHECK-NEXT:    typealias unsigned long size_t
// CHECK-NEXT:    template class allocator
// CHECK-NEXT:      parameter typename<0, 0> T
// CHECK-NEXT:      typealias T value_type
// CHECK-NEXT:      template class rebind
// CHECK-NEXT:        parameter typename<0, 1> U
// CHECK-NEXT:        typealias allocator<U> other
// CHECK-NEXT:      function T* allocate(unsigned long)
// CHECK-NEXT:      function void deallocate(T*, unsigned long)
// CHECK-NEXT:      [specializations]
// CHECK-NEXT:        class allocator<int>
// CHECK-NEXT:          typealias int value_type
// CHECK-NEXT:          template class rebind
// CHECK-NEXT:            parameter typename<0, 1> U
// CHECK-NEXT:            typealias allocator<U> other
// CHECK-NEXT:            [specializations]
// CHECK-NEXT:              class rebind<char>
// CHECK-NEXT:                typealias allocator<U> other
// CHECK-NEXT:              class rebind<short>
// CHECK-NEXT:                typealias allocator<U> other
// CHECK-NEXT:          function int* allocate(unsigned long)
// CHECK-NEXT:          function void deallocate(int*, unsigned long)
// CHECK-NEXT:  variable std::allocator<int>::rebind<char> alloc8
// CHECK-NEXT:  variable std::allocator<int>::rebind<short> alloc16