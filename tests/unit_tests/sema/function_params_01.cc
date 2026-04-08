// clang-format off
// RUN: %cxx -verify -dump-symbols %s | %filecheck %s
// clang-format on

struct C {
  void aa() const;
  void bb() volatile;
  void cc() const volatile;
  void dd() &;
  void ee() &&;
  void ff() const volatile& noexcept;
};

// clang-format off
//      CHECK:namespace
// CHECK-NEXT:  class C
// CHECK-NEXT:    constructor defaulted void C()
// CHECK-NEXT:    constructor defaulted void C(const ::C&)
// CHECK-NEXT:    constructor defaulted void C(::C&&)
// CHECK-NEXT:    injected class name C
// CHECK-NEXT:    function void aa() const
// CHECK-NEXT:    function void bb() volatile
// CHECK-NEXT:    function void cc() const volatile
// CHECK-NEXT:    function void dd() &
// CHECK-NEXT:    function void ee() &&
// CHECK-NEXT:    function void ff() const volatile & noexcept
// CHECK-NEXT:    function defaulted ::C& operator =(const ::C&)
// CHECK-NEXT:    function defaulted ::C& operator =(::C&&)
// CHECK-NEXT:    function defaulted void ~C()