// RUN: %cxx -verify -fsyntax-only %s

void variable(void (*fp)(auto));  // expected-error {{a placeholder type is only allowed in a parameter of a function declaration or a lambda-expression}}

void nested_pointer(void (*)(auto));  // expected-error {{a placeholder type is only allowed in a parameter of a function declaration or a lambda-expression}}

void nested_named(int g(auto));  // expected-error {{a placeholder type is only allowed in a parameter of a function declaration or a lambda-expression}}

auto trailing() -> void (*)(auto);  // expected-error {{a placeholder type is only allowed in a parameter of a function declaration or a lambda-expression}}

template <class T>
void with_template(void (*)(auto, T));  // expected-error {{a placeholder type is only allowed in a parameter of a function declaration or a lambda-expression}}

void well_formed(auto x, void (*callback)(int));
void well_formed_pack(auto x, auto... ys);
void (*returns_pointer(auto x))(int);
struct S {
  void member(auto) const;
  S(auto);
};
