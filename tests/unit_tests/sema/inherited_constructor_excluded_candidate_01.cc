// RUN: %cxx -verify -fsyntax-only %s

struct B {
  B() {}
  B(int) {}
};

struct M : B {
  using B::B;
};

// expected-note@+12 {{candidate constructor not viable: requires 0 arguments, but 1 was provided}}
// expected-note@+11 {{candidate constructor not viable: no known conversion from 'B' to 'const D&' for argument 1}}
// expected-note@+10 {{candidate constructor not viable: no known conversion from 'B' to 'D&&' for argument 1}}
// expected-note@+9 {{candidate constructor not viable: no known conversion from 'B' to 'int' for argument 1}}
// expected-note@+8 {{candidate constructor not viable: requires 0 arguments, but 1 was provided}}
// expected-note@+7 {{candidate constructor not viable: no known conversion from 'M' to 'const D&' for argument 1}}
// expected-note@+6 {{candidate constructor not viable: no known conversion from 'M' to 'D&&' for argument 1}}
// expected-note@+5 {{candidate constructor not viable: no known conversion from 'M' to 'int' for argument 1}}
// expected-note@+4 {{candidate constructor not viable: requires 0 arguments, but 1 was provided}}
// expected-note@+3 {{candidate constructor not viable: no known conversion from 'Z' to 'const D&' for argument 1}}
// expected-note@+2 {{candidate constructor not viable: no known conversion from 'Z' to 'D&&' for argument 1}}
// expected-note@+1 {{candidate constructor not viable: no known conversion from 'Z' to 'int' for argument 1}}
struct D : M {
  using M::M;
};

struct Z {};

void chained() {
  B b;
  M m;
  Z z;

  D fromInt(1);

  D fromBase(b);  // expected-error {{no matching constructor for initialization of 'D'}}
  D fromMid(m);   // expected-error {{no matching constructor for initialization of 'D'}}
  D fromZ(z);     // expected-error {{no matching constructor for initialization of 'D'}}
}

// expected-note@+2 {{candidate constructor not viable: no known conversion from 'B' to 'const TakesD&' for argument 1}}
// expected-note@+1 {{candidate constructor not viable: no known conversion from 'B' to 'TakesD&&' for argument 1}}
struct TakesD {
  // expected-note@+1 {{candidate constructor not viable: no known conversion from 'B' to 'const D&' for argument 1}}
  TakesD(const D&) {}
};

void throughConversion() {
  B b;
  TakesD t(b);  // expected-error {{no matching constructor for initialization of 'TakesD'}}
}

struct N : B {
  using B::B;
  N(const B&) {}
};

struct E : N {
  using N::N;
};

void ownConverting() {
  B b;
  E e(b);
}
