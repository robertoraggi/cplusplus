// RUN: %cxx -std=c++26 -verify -fsyntax-only %s
// expected-no-diagnostics

template <class...>
using void_t = void;

template <class D, class = void>
struct Type2 {
  static constexpr bool value = false;
};
template <class D>
struct Type2<D, void_t<typename D::A::B>> {
  static constexpr bool value = true;
};

template <class D, class = void>
struct Type3 {
  static constexpr bool value = false;
};
template <class D>
struct Type3<D, void_t<typename D::A::B::C>> {
  static constexpr bool value = true;
};

template <class D, class = void>
struct Value2 {
  static constexpr bool value = false;
};
template <class D>
struct Value2<D, void_t<decltype(D::A::v)>> {
  static constexpr bool value = true;
};

template <class D, class = void>
struct Tmpl2 {
  static constexpr bool value = false;
};
template <class D>
struct Tmpl2<D, void_t<typename D::A::template T<int>>> {
  static constexpr bool value = true;
};

template <class D, class = void>
struct TmplQual {
  static constexpr bool value = false;
};
template <class D>
struct TmplQual<D, void_t<typename D::template W<int>::B>> {
  static constexpr bool value = true;
};

struct NoA {};
struct HasAB {
  struct A {
    using B = int;
  };
};
struct HasABC {
  struct A {
    struct B {
      using C = int;
    };
  };
};
struct HasABNoC {
  struct A {
    struct B {};
  };
};
struct DerivedAB : HasAB {};
struct Left {
  struct A {
    using B = int;
  };
};
struct Right {
  struct A {
    using B = long;
  };
};
struct Ambiguous : Left, Right {};
class PrivateA {
  struct A {
    using B = int;
  };
};
struct StaticA {
  static int A;
};
struct FunctionA {
  void A();
};
struct EnumA {
  enum A { B };
};
struct ScopedEnumA {
  enum class A { B };
};
struct TemplateA {
  template <class>
  struct A {
    using B = int;
  };
};
struct Inner {
  using B = int;
};
struct AliasA {
  using A = Inner;
};
struct AliasToInt {
  using A = int;
};
template <class T>
struct Wrap {
  struct A {
    using B = T;
  };
};
template <class T>
struct Spec {
  struct A {
    using B = T;
  };
};
template <>
struct Spec<int> {};
struct ValueA {
  struct A {
    static constexpr int v = 1;
  };
};
struct TypeOnlyA {
  struct A {};
};
struct HasTmpl {
  struct A {
    template <class>
    using T = int;
  };
};
struct HasW {
  template <class>
  struct W {
    using B = int;
  };
};
struct HasWNoB {
  template <class>
  struct W {};
};
enum Plain { A };

static_assert(Type2<HasAB>::value);
static_assert(!Type2<NoA>::value);
static_assert(Type2<DerivedAB>::value);
static_assert(!Type2<Ambiguous>::value);
static_assert(!Type2<PrivateA>::value);
static_assert(!Type2<StaticA>::value);
static_assert(!Type2<FunctionA>::value);
static_assert(!Type2<EnumA>::value);
static_assert(!Type2<ScopedEnumA>::value);
static_assert(!Type2<TemplateA>::value);
static_assert(Type2<AliasA>::value);
static_assert(!Type2<AliasToInt>::value);
static_assert(Type2<Wrap<char>>::value);
static_assert(Type2<Spec<char>>::value);
static_assert(!Type2<Spec<int>>::value);
static_assert(!Type2<int>::value);
static_assert(!Type2<Plain>::value);
static_assert(!Type2<TypeOnlyA>::value);

static_assert(Type3<HasABC>::value);
static_assert(!Type3<HasABNoC>::value);
static_assert(!Type3<HasAB>::value);
static_assert(!Type3<NoA>::value);

static_assert(Value2<ValueA>::value);
static_assert(!Value2<NoA>::value);
static_assert(!Value2<TypeOnlyA>::value);

static_assert(Tmpl2<HasTmpl>::value);
static_assert(!Tmpl2<NoA>::value);
static_assert(!Tmpl2<TypeOnlyA>::value);

static_assert(TmplQual<HasW>::value);
static_assert(!TmplQual<HasWNoB>::value);
static_assert(!TmplQual<NoA>::value);

template <class D>
concept Deep = requires { typename D::A::B; };
template <class D>
concept DeepValue = requires { D::A::v; };

static_assert(Deep<HasAB>);
static_assert(!Deep<NoA>);
static_assert(Deep<DerivedAB>);
static_assert(!Deep<PrivateA>);
static_assert(DeepValue<ValueA>);
static_assert(!DeepValue<NoA>);
static_assert(!DeepValue<TypeOnlyA>);

template <class D>
  requires requires { typename D::A::B; }
constexpr int pick(D) {
  return 1;
}
constexpr int pick(...) { return 2; }
static_assert(pick(HasAB{}) == 1);
static_assert(pick(NoA{}) == 2);

template <class D>
constexpr auto viaReturn(D) -> typename D::A::B {
  return 1;
}
constexpr int viaReturn(...) { return 2; }
static_assert(viaReturn(HasAB{}) == 1);
static_assert(viaReturn(NoA{}) == 2);

template <class D>
constexpr auto viaDecltype(D) -> decltype(D::A::v) {
  return 1;
}
constexpr int viaDecltype(...) { return 2; }
static_assert(viaDecltype(ValueA{}) == 1);
static_assert(viaDecltype(NoA{}) == 2);

template <class D>
struct Current {
  struct A {
    using B = D;
  };
  using Self = Current;
  using Got = typename Self::A::B;
};
static_assert(__is_same(Current<int>::Got, int));
