// clang-format off
// RUN: %cxx -verify -fcheck -dump-symbols %s | %filecheck %s

struct Base {
  virtual void f();
  virtual void g() = 0;
  virtual void h();
  virtual ~Base() = default;
};

// CHECK:      class Base polymorphic abstract
// CHECK:        function virtual void f()
// CHECK:        function virtual pure void g()
// CHECK:        function virtual void h()
// CHECK:        function inline virtual defaulted void ~Base()

struct Derived : Base {
  void f() override;
  void g() override final;
  void h() override;
  ~Derived() override;
};

// CHECK:      class Derived polymorphic
// CHECK:        function virtual override void f()
// CHECK:        function virtual override final void g()
// CHECK:        function virtual override void h()
// CHECK:        function virtual override void ~Derived()

struct ImplicitVirtual : Base {
  void f();            // implicitly virtual
  void g();            // implicitly virtual
  void h();            // implicitly virtual
};

// CHECK:      class ImplicitVirtual polymorphic
// CHECK:        function virtual void f()
// CHECK:        function virtual void g()
// CHECK:        function virtual void h()

struct PartialOverride : Base {
  void f() override;   // overrides f, g still pure
};

// CHECK:      class PartialOverride polymorphic abstract

struct A {
  virtual void x() = 0;
  virtual void y() = 0;
};

struct B : A {
  void x() override {}  // overrides x; y still pure
};

struct C : B {
  void y() override {}  // overrides y; all resolved
};

struct D : C {};  // inherits from concrete C; not abstract

// CHECK:      class A polymorphic abstract
// CHECK:      class B polymorphic abstract
// CHECK:      class C polymorphic
// CHECK:      class D polymorphic

struct VBase {
  virtual ~VBase() = default;
};

struct VDerived : VBase {};

struct VGrandchild : VDerived {};

// CHECK:      class VBase polymorphic
// CHECK:        function inline virtual defaulted void ~VBase()
// CHECK:      class VDerived polymorphic
// CHECK:        function virtual defaulted void ~VDerived()
// CHECK:      class VGrandchild polymorphic
// CHECK:        function virtual defaulted void ~VGrandchild()

struct Plain {
  void f();
  int x;
};

// CHECK:      class Plain{{$}}

static_assert(__is_polymorphic(Base), "Base is polymorphic");
static_assert(__is_polymorphic(Derived), "Derived is polymorphic");
static_assert(__is_polymorphic(ImplicitVirtual), "ImplicitVirtual is polymorphic");
static_assert(__is_polymorphic(VDerived), "VDerived is polymorphic");
static_assert(__is_polymorphic(VGrandchild), "VGrandchild is polymorphic");
static_assert(!__is_polymorphic(Plain), "Plain is not polymorphic");
static_assert(!__is_polymorphic(int), "int is not polymorphic");

static_assert(__is_abstract(Base), "Base is abstract");
static_assert(__is_abstract(PartialOverride), "PartialOverride is abstract");
static_assert(__is_abstract(B), "B is abstract");
static_assert(!__is_abstract(C), "C is not abstract");
static_assert(!__is_abstract(D), "D is not abstract");
static_assert(!__is_abstract(Derived), "Derived is not abstract");
static_assert(!__is_abstract(Plain), "Plain is not abstract");

static_assert(__has_virtual_destructor(Base), "Base has virtual destructor");
static_assert(__has_virtual_destructor(Derived), "Derived has virtual destructor");
static_assert(__has_virtual_destructor(VBase), "VBase has virtual destructor");
static_assert(__has_virtual_destructor(VDerived), "VDerived has virtual destructor");
static_assert(__has_virtual_destructor(VGrandchild), "VGrandchild has virtual destructor");
static_assert(!__has_virtual_destructor(Plain), "Plain has no virtual destructor");
static_assert(!__has_virtual_destructor(A), "A has no virtual destructor");
