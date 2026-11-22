// RUN: %cxx -verify -fsyntax-only %s

virtual int x;  // expected-error {{'virtual' can be used only in a function declaration}}
_Noreturn int v;  // expected-error {{'_Noreturn' can be used only in a function declaration}}
explicit(false) int e;  // expected-error {{'explicit' can be used only in a function declaration}}
typedef virtual int T;  // expected-error {{'virtual' can be used only in a function declaration}}
typedef _Noreturn void F();  // expected-error {{'_Noreturn' can be used only in a function declaration}}
void f(virtual int p);  // expected-error {{'virtual' can be used only in a function declaration}}
virtual struct X { int i; };  // expected-error {{'virtual' can be used only in a function declaration}}

struct S {
  virtual int m;  // expected-error {{'virtual' can be used only in a function declaration}}
  explicit int n;  // expected-error {{'explicit' can be used only in a function declaration}}
};

template <bool B>
struct Z {
  explicit(B) int m;  // expected-error {{'explicit' can be used only in a function declaration}}
};

Z<true> z;

typedef void G();
_Noreturn G g;
