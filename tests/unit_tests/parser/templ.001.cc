// RUN: %cxx -verify -dump-symbols %s -o - | %filecheck %s

template <typename T>
struct is_void {
  enum { value = __is_void(T) };
};

// CHECK-NEXT: - namespace:{{$}}
// CHECK-NEXT: - template struct: is_void{{$}}
// CHECK-NEXT: - template parameters:{{$}}
// CHECK-NEXT: - type parameter: T{{$}}
// CHECK-NEXT: - enum:{{$}}
// CHECK-NEXT: - enumerator: value{{$}}
