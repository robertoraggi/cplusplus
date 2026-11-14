// RUN: %cxx -toolchain macos -std=c++26 -verify -fsyntax-only %s
// expected-no-diagnostics

struct E {};
struct G {};

struct Big {
  [[no_unique_address]] G a;
  [[no_unique_address]] G b;
  [[no_unique_address]] G c;
  [[no_unique_address]] G d;
  [[no_unique_address]] G e;
  [[no_unique_address]] G f;
  [[no_unique_address]] G g;
  [[no_unique_address]] G h;
  [[no_unique_address]] G i;
  [[no_unique_address]] G j;
};

struct Two {
  [[no_unique_address]] G a;
  [[no_unique_address]] G b;
};

struct OneEmptyMember : virtual E {
  [[no_unique_address]] G m;
};
struct LargeEmptyMember : virtual E {
  [[no_unique_address]] Big m;
};
struct SmallEmptyMember : virtual E {
  [[no_unique_address]] Two m;
};
struct TwoEmptyMembers : virtual E {
  [[no_unique_address]] G m;
  [[no_unique_address]] G n;
};
struct EmptyMemberAfterBase : G, virtual E {
  [[no_unique_address]] G m;
};
struct LargeEmptyBase : Big, virtual E {};
struct MemberOfVirtualBaseType : virtual E {
  [[no_unique_address]] E m;
};
struct ArrayMember : virtual E {
  [[no_unique_address]] G m[2];
};
struct ZeroWidthBitField : virtual E {
  int : 0;
  [[no_unique_address]] G m;
};

template <typename Base>
struct Derived : virtual Base {
  char c;
};

constexpr auto pointer = sizeof(void*);

static_assert(sizeof(Derived<OneEmptyMember>) == 2 * pointer);
static_assert(sizeof(Derived<LargeEmptyMember>) == 4 * pointer);
static_assert(sizeof(Derived<SmallEmptyMember>) == 2 * pointer);
static_assert(sizeof(Derived<TwoEmptyMembers>) == 4 * pointer);
static_assert(sizeof(Derived<EmptyMemberAfterBase>) == 4 * pointer);
static_assert(sizeof(Derived<LargeEmptyBase>) == 4 * pointer);
static_assert(sizeof(Derived<MemberOfVirtualBaseType>) == 2 * pointer);
static_assert(sizeof(Derived<ArrayMember>) == 4 * pointer);
static_assert(sizeof(Derived<ZeroWidthBitField>) == 2 * pointer);
