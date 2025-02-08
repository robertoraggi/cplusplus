// RUN: %cxx -verify -fcheck %s

struct X {
  void f() { static_assert(__is_same(decltype(this), X*)); }

  void f() const { static_assert(__is_same(decltype(this), const X*)); }

  void f() volatile { static_assert(__is_same(decltype(this), volatile X*)); }

  void f() const volatile {
    static_assert(__is_same(decltype(this), const volatile X*));
  }

  void g();
  void g() const;
  void g() volatile;
  void g() const volatile;
};

void X::g() { static_assert(__is_same(decltype(this), X*)); }

void X::g() const { static_assert(__is_same(decltype(this), const X*)); }

void X::g() volatile { static_assert(__is_same(decltype(this), volatile X*)); }

void X::g() const volatile {
  static_assert(__is_same(decltype(this), const volatile X*));
}
