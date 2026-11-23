// RUN: %cxx -std=c++26 -verify -fsyntax-only %s

struct NoA {};
namespace N {}
class PrivateA {
  struct A {
    using B = int;
  };
};
struct TemplateA {
  template <class>
  struct A {
    using B = int;
  };
};
struct Left {
  struct A {
    using B = int;
  };
};
struct Right {
  struct A {
    using B = int;
  };
};
struct Both : Left, Right {};
using Int = int;
int variable = 0;

NoA::A::B missingType;  // expected-error {{no member named 'A' in 'NoA'}}
typename NoA::A::B missingTypename;  // expected-error {{no member named 'A' in 'NoA'}}
int missingValue = NoA::A::v;  // expected-error {{no member named 'A' in 'NoA'}}
N::A::B missingNamespaceMember;  // expected-error {{no member named 'A' in 'N'}}
void NoA::A::f() {}  // expected-error {{no member named 'A' in 'NoA'}}

Unknown::B undeclared;  // expected-error {{lookup for 'Unknown' finds no declaration}}
int undeclaredValue = Unknown::v;  // expected-error {{lookup for 'Unknown' finds no declaration}}
void Unknown::f() {}  // expected-error {{lookup for 'Unknown' finds no declaration}}
Nope<int>::X undeclaredTemplate;  // expected-error {{lookup for 'Nope' finds no declaration}}
int variableValue = variable::k;  // expected-error {{lookup for 'variable' finds no declaration}}

PrivateA::A::B privateType;  // expected-error {{'A' is a private member of 'PrivateA'}}
typename PrivateA::A::B privateTypename;  // expected-error {{'A' is a private member of 'PrivateA'}}
PrivateA::A privateClass;  // expected-error {{'A' is a private member of 'PrivateA'}}

Both::A::B ambiguousType;  // expected-error {{reference to 'A' is ambiguous}}
typename Both::A::B ambiguousTypename;  // expected-error {{reference to 'A' is ambiguous}}

TemplateA::A::B templateName;  // expected-error {{template 'A' cannot be used prior to '::' without a template argument list}}
int templateValue = sizeof(TemplateA::A::B);  // expected-error {{template 'A' cannot be used prior to '::' without a template argument list}}

Int::x aliasOfInt;  // expected-error {{type 'int' cannot be used prior to '::' because it has no members}}
int aliasValue = Int::v;  // expected-error {{type 'int' cannot be used prior to '::' because it has no members}}
decltype(1)::x decltypeOfInt;  // expected-error {{type 'int' cannot be used prior to '::' because it has no members}}
int decltypeValue = decltype(1)::v;  // expected-error {{type 'int' cannot be used prior to '::' because it has no members}}

struct S {
  static const int v = 1;
};

int memberQualifier(S s) {
  return s.Base::v;  // expected-error {{lookup for 'Base' finds no declaration}}
}

template <class T>
int dependentMemberQualifier(T t) {
  return t.Base::v;  // expected-error {{lookup for 'Base' finds no declaration}}
}

int instantiated = dependentMemberQualifier(S{});  // expected-note {{in instantiation of function template specialization 'dependentMemberQualifier<S>' requested here}}

template <class D>
struct UseNestedTemplate {
  typename D::A::template T<int>::B x;  // expected-error {{no member named 'A' in 'NoA'}}
};

UseNestedTemplate<NoA> useNestedTemplate;  // expected-note {{in instantiation of template class 'UseNestedTemplate<NoA>' requested here}}

template <class D>
struct UseMemberTemplate {
  typename D::template W<int>::B x;  // expected-error {{no member named 'W' in 'NoA'}}
};

UseMemberTemplate<NoA> useMemberTemplate;  // expected-note {{in instantiation of template class 'UseMemberTemplate<NoA>' requested here}}

template <class T>
int dependentTemplateQualifier(T t) {
  return t.template Nope<int>::v;  // expected-error {{lookup for 'Nope' finds no declaration}}
}

int instantiatedTemplate = dependentTemplateQualifier(S{});  // expected-note {{in instantiation of function template specialization 'dependentTemplateQualifier<S>' requested here}}

struct Conversion {
  using T = int;
  operator T();
};

Conversion::operator Q() { return 0; }  // expected-error {{lookup for 'Q' finds no declaration}}
