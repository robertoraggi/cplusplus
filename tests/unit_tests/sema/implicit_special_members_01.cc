// RUN: %cxx -fsyntax-only -dump-symbols %s | %filecheck %s --match-full-lines

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
// CHECK-NEXT:    constructor defaulted void Pod() noexcept
// CHECK-NEXT:    constructor defaulted void Pod(const ::Pod&) noexcept
// CHECK-NEXT:      parameters
// CHECK-NEXT:        parameter const ::Pod&
// CHECK-NEXT:    constructor defaulted void Pod(::Pod&&) noexcept
// CHECK-NEXT:      parameters
// CHECK-NEXT:        parameter ::Pod&&
// CHECK-NEXT:    injected class name Pod
// CHECK-NEXT:    field int x
// CHECK-NEXT:    field float y
// CHECK-NEXT:    function defaulted ::Pod& operator =(const ::Pod&) noexcept
// CHECK-NEXT:      parameters
// CHECK-NEXT:        parameter const ::Pod&
// CHECK-NEXT:    function defaulted ::Pod& operator =(::Pod&&) noexcept
// CHECK-NEXT:      parameters
// CHECK-NEXT:        parameter ::Pod&&
// CHECK-NEXT:    function defaulted void ~Pod() noexcept
// CHECK-NEXT:  class WithCtor
// CHECK-NEXT:    constructor inline void WithCtor()
// CHECK-NEXT:      block
// CHECK-NEXT:        variable static constexpr const char __func__[9]
// CHECK-NEXT:    constructor defaulted void WithCtor(const ::WithCtor&) noexcept
// CHECK-NEXT:      parameters
// CHECK-NEXT:        parameter const ::WithCtor&
// CHECK-NEXT:    constructor defaulted void WithCtor(::WithCtor&&) noexcept
// CHECK-NEXT:      parameters
// CHECK-NEXT:        parameter ::WithCtor&&
// CHECK-NEXT:    injected class name WithCtor
// CHECK-NEXT:    field int val
// CHECK-NEXT:    function defaulted ::WithCtor& operator =(const ::WithCtor&) noexcept
// CHECK-NEXT:      parameters
// CHECK-NEXT:        parameter const ::WithCtor&
// CHECK-NEXT:    function defaulted ::WithCtor& operator =(::WithCtor&&) noexcept
// CHECK-NEXT:      parameters
// CHECK-NEXT:        parameter ::WithCtor&&
// CHECK-NEXT:    function defaulted void ~WithCtor() noexcept
// CHECK-NEXT:  class WithDtor
// CHECK-NEXT:    constructor defaulted void WithDtor() noexcept
// CHECK-NEXT:    constructor defaulted void WithDtor(const ::WithDtor&) noexcept
// CHECK-NEXT:      parameters
// CHECK-NEXT:        parameter const ::WithDtor&
// CHECK-NEXT:    injected class name WithDtor
// CHECK-NEXT:    function inline void ~WithDtor() noexcept
// CHECK-NEXT:      block
// CHECK-NEXT:        variable static constexpr const char __func__[10]
// CHECK-NEXT:    field int data
// CHECK-NEXT:    function defaulted ::WithDtor& operator =(const ::WithDtor&) noexcept
// CHECK-NEXT:      parameters
// CHECK-NEXT:        parameter const ::WithDtor&
