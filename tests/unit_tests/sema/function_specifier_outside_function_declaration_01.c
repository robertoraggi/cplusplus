// RUN: %cxx -verify -fsyntax-only %s

_Noreturn int v;  // expected-error {{'_Noreturn' can be used only in a function declaration}}
inline int a;  // expected-error {{'inline' can be used only in a function declaration}}
typedef _Noreturn void F(void);  // expected-error {{'_Noreturn' can be used only in a function declaration}}
typedef inline int T;  // expected-error {{'inline' can be used only in a function declaration}}
void f(_Noreturn int p);  // expected-error {{'_Noreturn' can be used only in a function declaration}}
struct W { inline int m; };  // expected-error {{'inline' can be used only in a function declaration}}

_Noreturn void ok(void);
inline void ok2(void) {}
