// clang-format off
// RUN: %cxx -verify -fcheck %s

struct NoBase {
  void f() override; // expected-error {{'f' marked 'override' but does not override any member function}}
};

struct Base {
  virtual void f();
  virtual void g() final;
};

struct GoodOverride : Base {
  void f() override;  // ok
};

struct BadOverrideFinal : Base {
  void g() override; // expected-error {{declaration of 'g' overrides a 'final' function}}
};

struct FinalClass final {};

struct DeriveFromFinal : FinalClass {}; // expected-error {{cannot derive from 'final' class 'FinalClass'}}
