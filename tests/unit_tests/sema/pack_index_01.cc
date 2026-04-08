// RUN: %cxx -dump-symbols %s | %filecheck %s
// expected-no-diagnostics

template <int i>
auto at(auto... xs) {
  return xs...[i];
}

int main() {
  int x = at<1>(1, 2, 3);
  return x - 2;
}

// clang-format off
//      CHECK: template function auto at(type-param<1, 0>...)
// CHECK-NEXT:   parameter constant<0, 0, int> i
// CHECK-NEXT:   parameter typename<1, 0>... __auto_1
// CHECK-NEXT:   parameters
// CHECK-NEXT:     parameter type-param<1, 0>... xs
// CHECK-NEXT:     block
// CHECK-NEXT:       variable static constexpr const char __func__[3]
// CHECK-NEXT:   [specializations]
// CHECK-NEXT:     function int at(int, int, int)
