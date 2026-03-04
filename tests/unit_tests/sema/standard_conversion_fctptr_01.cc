// RUN: %cxx -verify -fcheck %s

// clang-format off

// expected-no-diagnostics

void f_noexcept(int) noexcept;

void test_strip_noexcept_free_function() {
  void (*fp)(int) = &f_noexcept;
}

void test_strip_noexcept_implicit() {
  void (*fp)(int) = f_noexcept;
}

struct S {
  void g(int) noexcept;
};

void test_strip_noexcept_member_function() {
  void (S::* mp)(int) = &S::g;
}
