// clang-format off
// RUN: %cxx -verify -fcheck -dump-symbols %s | %filecheck %s
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
// CHECK: class C
// CHECK:   function void aa() const
// CHECK:   function void bb() volatile
// CHECK:   function void cc() const volatile
// CHECK:   function void dd() &
// CHECK:   function void ee() &&
// CHECK:   function void ff() const volatile & noexcept