// RUN: %cxx -verify -fcheck %s

struct Empty {};

struct NonEmpty {
  int x;
};

struct Base {
  virtual void f();
};

struct Derived : Base {};

struct Final final : Base {};

struct Abstract {
  virtual void f() = 0;
};

struct ConcreteFromAbstract : Abstract {
  void f() override {}
};

struct VirtualDtor {
  virtual ~VirtualDtor() = default;
};

struct NonVirtualDtor {
  ~NonVirtualDtor() = default;
};

struct InheritedVirtualDtor : VirtualDtor {};

struct Trivial {};

struct NonTrivial {
  NonTrivial() {}
};

struct VirtualBase : virtual Base {};

struct Aggregate {
  int x;
  double y;
};

struct NonAggregate {
  NonAggregate(int) {}
};

union MyUnion {
  int i;
  float f;
};

enum E { A, B };
enum class SE { X, Y };

struct MultipleVirtual {
  virtual void f();
  virtual void g();
};

struct PureMultiple {
  virtual void f() = 0;
  virtual void g() = 0;
};

//
// __is_class
//

static_assert(__is_class(Empty));
static_assert(__is_class(Base));
static_assert(__is_class(Derived));
// expected-error@1 {{static assert failed}}
static_assert(__is_class(int));
// expected-error@1 {{static assert failed}}
static_assert(__is_class(MyUnion));
// expected-error@1 {{static assert failed}}
static_assert(__is_class(E));

//
// __is_union
//

static_assert(__is_union(MyUnion));
// expected-error@1 {{static assert failed}}
static_assert(__is_union(Empty));
// expected-error@1 {{static assert failed}}
static_assert(__is_union(int));

//
// __is_enum
//

static_assert(__is_enum(E));
static_assert(__is_enum(SE));
// expected-error@1 {{static assert failed}}
static_assert(__is_enum(int));
// expected-error@1 {{static assert failed}}
static_assert(__is_enum(Empty));

//
// __is_scoped_enum
//

static_assert(__is_scoped_enum(SE));
// expected-error@1 {{static assert failed}}
static_assert(__is_scoped_enum(E));
// expected-error@1 {{static assert failed}}
static_assert(__is_scoped_enum(int));

//
// __is_final
//

static_assert(__is_final(Final));
// expected-error@1 {{static assert failed}}
static_assert(__is_final(Base));
// expected-error@1 {{static assert failed}}
static_assert(__is_final(Derived));
// expected-error@1 {{static assert failed}}
static_assert(__is_final(int));

//
// __is_empty
//

static_assert(__is_empty(Empty));
// expected-error@1 {{static assert failed}}
static_assert(__is_empty(NonEmpty));
// expected-error@1 {{static assert failed}}
static_assert(__is_empty(Base));  // has virtual function
// expected-error@1 {{static assert failed}}
static_assert(__is_empty(int));

//
// __is_polymorphic
//

static_assert(__is_polymorphic(Base));
static_assert(__is_polymorphic(Derived));
static_assert(__is_polymorphic(Abstract));
static_assert(__is_polymorphic(VirtualDtor));
// expected-error@1 {{static assert failed}}
static_assert(__is_polymorphic(Empty));
// expected-error@1 {{static assert failed}}
static_assert(__is_polymorphic(Trivial));
// expected-error@1 {{static assert failed}}
static_assert(__is_polymorphic(int));

//
// __is_abstract
//

static_assert(__is_abstract(Abstract));
static_assert(__is_abstract(PureMultiple));
// expected-error@1 {{static assert failed}}
static_assert(__is_abstract(Base));
// expected-error@1 {{static assert failed}}
static_assert(__is_abstract(ConcreteFromAbstract));
// expected-error@1 {{static assert failed}}
static_assert(__is_abstract(Empty));
// expected-error@1 {{static assert failed}}
static_assert(__is_abstract(int));

//
// __has_virtual_destructor
//

static_assert(__has_virtual_destructor(VirtualDtor));
static_assert(__has_virtual_destructor(InheritedVirtualDtor));
// expected-error@1 {{static assert failed}}
static_assert(__has_virtual_destructor(NonVirtualDtor));
// expected-error@1 {{static assert failed}}
static_assert(__has_virtual_destructor(Empty));
// expected-error@1 {{static assert failed}}
static_assert(__has_virtual_destructor(int));

//
// __is_trivial
//

static_assert(__is_trivial(int));
static_assert(__is_trivial(Trivial));
static_assert(__is_trivial(Empty));
// expected-error@1 {{static assert failed}}
static_assert(__is_trivial(NonTrivial));
// expected-error@1 {{static assert failed}}
static_assert(__is_trivial(Base));  // has virtual function

//
// __is_trivially_constructible
//

static_assert(__is_trivially_constructible(int));
static_assert(__is_trivially_constructible(Trivial));
static_assert(__is_trivially_constructible(Empty));
// expected-error@1 {{static assert failed}}
static_assert(__is_trivially_constructible(NonTrivial));
// expected-error@1 {{static assert failed}}
static_assert(__is_trivially_constructible(Base));

//
// __is_standard_layout
//

static_assert(__is_standard_layout(int));
static_assert(__is_standard_layout(Empty));
static_assert(__is_standard_layout(NonEmpty));
static_assert(__is_standard_layout(Aggregate));
// expected-error@1 {{static assert failed}}
static_assert(__is_standard_layout(Base));  // has virtual function
// expected-error@1 {{static assert failed}}
static_assert(__is_standard_layout(VirtualBase));  // has virtual base

//
// __is_pod
//

static_assert(__is_pod(int));
static_assert(__is_pod(Empty));
static_assert(__is_pod(Aggregate));
// expected-error@1 {{static assert failed}}
static_assert(__is_pod(NonTrivial));
// expected-error@1 {{static assert failed}}
static_assert(__is_pod(Base));

//
// __is_literal_type
//

static_assert(__is_literal_type(int));
static_assert(__is_literal_type(void));
static_assert(__is_literal_type(Empty));
static_assert(__is_literal_type(int&));

//
// __is_aggregate
//

static_assert(__is_aggregate(Aggregate));
static_assert(__is_aggregate(int[10]));
static_assert(__is_aggregate(Empty));
// expected-error@1 {{static assert failed}}
static_assert(__is_aggregate(NonAggregate));
// expected-error@1 {{static assert failed}}
static_assert(__is_aggregate(Base));  // has virtual function
// expected-error@1 {{static assert failed}}
static_assert(__is_aggregate(int));

//
// __is_trivially_assignable
//

static_assert(__is_trivially_assignable(int&, int));
// expected-error@1 {{static assert failed}}
static_assert(__is_trivially_assignable(Base&, Base));

//
// __is_base_of
//

static_assert(__is_base_of(Base, Derived));
static_assert(__is_base_of(Base, Final));
static_assert(__is_base_of(Base, Base));
// expected-error@1 {{static assert failed}}
static_assert(__is_base_of(Derived, Base));
// expected-error@1 {{static assert failed}}
static_assert(__is_base_of(int, int));

//
// __is_same
//

static_assert(__is_same(int, int));
static_assert(__is_same(const int, const int));
// expected-error@1 {{static assert failed}}
static_assert(__is_same(int, const int));
// expected-error@1 {{static assert failed}}
static_assert(__is_same(int, float));
