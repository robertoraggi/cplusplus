// clang-format off
// RUN: %cxx -verify -fcheck -dump-symbols %s | %filecheck %s
// clang-format on

struct C {
  void f();
  auto f(int) -> int;
  auto f(const int&) -> const int&;
  auto f(double) -> double;

  struct iterator;
  auto begin() -> iterator;
  auto end() -> iterator;

  struct const_iterator;
  auto begin() const -> const_iterator;
  auto end() const -> const_iterator;

  auto operator=(const C&) -> C&;
  auto operator=(C&&) -> C&;
};

// clang-format off
// CHECK:namespace
// CHECK:  class C
// CHECK:    function void f()
// CHECK:    function int f(int)
// CHECK:    function const int& f(const int&)
// CHECK:    function double f(double)
// CHECK:    class iterator
// CHECK:    function iterator begin()
// CHECK:    function const_iterator begin() const
// CHECK:    function iterator end()
// CHECK:    function const_iterator end() const
// CHECK:    class const_iterator
// CHECK:    function C& operator =(const C&)
// CHECK:    function C& operator =(C&&)