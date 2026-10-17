// RUN: %cxx -verify -fsyntax-only %s

struct A {
  enum class E : int;
  enum U : int;
  enum class Twice : int { t };
};

struct Base {
  enum class Inherited : int { i };
};

struct Derived : Base {};

namespace N {
enum class InN : int;
struct S {
  enum class E : int;
};
}  // namespace N

// expected-error@+1 {{enumeration 'Missing' is not declared in 'A'}}
enum class A::Missing : int { m };

// expected-error@+1 {{enumeration 'E' redeclared with different underlying type}}
enum class A::E : char { e };

// expected-error@+1 {{enumeration 'E2' is not declared in 'A'}}
enum A::E2 : int { e2 };

// expected-error@+1 {{enumeration 'U' redeclared with different scopedness}}
enum class A::U : int { u };

// expected-error@+1 {{redefinition of enumeration 'Twice'}}
enum class A::Twice : int { again };

// expected-error@+1 {{enumeration 'Inherited' is not declared in 'Derived'}}
enum class Derived::Inherited : int { j };

// expected-error@+1 {{enumeration 'Missing2' is not declared in 'N'}}
enum class N::Missing2 : int { x };

enum class N::InN : int { a };

// expected-error@+1 {{redefinition of enumeration 'InN'}}
enum class N::InN : int { b };

namespace M {
// expected-error@+1 {{enumeration 'InN' shall be declared in a namespace enclosing 'N'}}
enum class N::InN : int { c };
}  // namespace M

namespace N::Inner {
// expected-error@+1 {{enumeration 'E' shall be declared in a namespace enclosing 'S'}}
enum class S::E : int { d };
}  // namespace N::Inner

struct W {
  enum class Q : int;
};

void function() {
  // expected-error@+1 {{enumeration 'Q' with a nested-name-specifier shall be declared in a namespace scope}}
  enum class W::Q : int { q };
}

struct Holder {
  // expected-error@+1 {{enumeration 'Q' with a nested-name-specifier shall not be declared in a class scope}}
  enum class W::Q : int { r };
};
