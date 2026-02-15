// RUN: %cxx -fcheck -dump-symbols %s | %filecheck %s --match-full-lines

struct Pod {
  int x;
  float y;
};

struct WithCtor {
  WithCtor() {}
  int val;
};

struct WithDtor {
  ~WithDtor() {}
  int data;
};

// clang-format off
//      CHECK:namespace
// CHECK-NEXT:  class Pod
// CHECK-NEXT:    constructor defaulted void Pod()
// CHECK-NEXT:    constructor defaulted void Pod(const ::Pod&)
// CHECK-NEXT:    constructor defaulted void Pod(::Pod&&)
// CHECK-NEXT:    field int x
// CHECK-NEXT:    field float y
// CHECK-NEXT:    function defaulted ::Pod& operator =(const ::Pod&)
// CHECK-NEXT:    function defaulted ::Pod& operator =(::Pod&&)
// CHECK-NEXT:    function defaulted void ~Pod()
// CHECK-NEXT:  class WithCtor
// CHECK-NEXT:    constructor inline void WithCtor()
// CHECK-NEXT:      block
// CHECK-NEXT:        variable static constexpr const char __func__[9]
// CHECK-NEXT:    constructor defaulted void WithCtor(const ::WithCtor&)
// CHECK-NEXT:    constructor defaulted void WithCtor(::WithCtor&&)
// CHECK-NEXT:    field int val
// CHECK-NEXT:    function defaulted ::WithCtor& operator =(const ::WithCtor&)
// CHECK-NEXT:    function defaulted ::WithCtor& operator =(::WithCtor&&)
// CHECK-NEXT:    function defaulted void ~WithCtor()
// CHECK-NEXT:  class WithDtor
// CHECK-NEXT:    constructor defaulted void WithDtor()
// CHECK-NEXT:    constructor defaulted void WithDtor(const ::WithDtor&)
// CHECK-NEXT:    constructor defaulted void WithDtor(::WithDtor&&)
// CHECK-NEXT:    function inline void ~WithDtor()
// CHECK-NEXT:      block
// CHECK-NEXT:        variable static constexpr const char __func__[10]
// CHECK-NEXT:    field int data
// CHECK-NEXT:    function defaulted ::WithDtor& operator =(const ::WithDtor&)
// CHECK-NEXT:    function defaulted ::WithDtor& operator =(::WithDtor&&)
