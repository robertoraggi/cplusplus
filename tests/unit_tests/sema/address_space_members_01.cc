// RUN: %cxx -toolchain wasm32 -verify -fsyntax-only -fvalidate-ast %s
// expected-no-diagnostics

using IntOne = int __attribute__((address_space(1)));
using IntTwo = int __attribute__((address_space(2)));

struct Prototype {
  auto self() -> decltype(this);
  auto self() const -> decltype(this);
  auto self() volatile -> decltype(this);
  auto self() const volatile -> decltype(this);
  IntOne* address() const;
  IntTwo* address() volatile;
  constexpr int select(IntOne*) const { return 1; }
  constexpr int select(IntTwo*) const { return 2; }
};
static_assert(__is_same(decltype(((Prototype*)nullptr)->self()), Prototype*));
static_assert(__is_same(decltype(((const Prototype*)nullptr)->self()),
                        const Prototype*));
static_assert(__is_same(decltype(((volatile Prototype*)nullptr)->self()),
                        volatile Prototype*));
static_assert(__is_same(decltype(((const volatile Prototype*)nullptr)->self()),
                        const volatile Prototype*));
static_assert(__is_same(decltype(((const Prototype*)nullptr)->address()),
                        IntOne*));
static_assert(__is_same(decltype(((volatile Prototype*)nullptr)->address()),
                        IntTwo*));
constexpr Prototype prototype{};
static_assert(prototype.select((IntOne*)nullptr) == 1);
static_assert(prototype.select((IntTwo*)nullptr) == 2);

template <class T>
concept HasImplicitSelf = requires(T& value) { value.self(); };
using PrototypeZero = Prototype __attribute__((address_space(0)));
using PrototypeOne = Prototype __attribute__((address_space(1)));
static_assert(!HasImplicitSelf<PrototypeZero>);
static_assert(!HasImplicitSelf<PrototypeOne>);
static_assert(!HasImplicitSelf<const PrototypeOne>);
static_assert(!HasImplicitSelf<volatile PrototypeOne>);
static_assert(!HasImplicitSelf<const volatile PrototypeOne>);

struct Record {
  int value;
  int array[2];
  constexpr int select(this const Record&) { return 0; }
  constexpr int select(this const Record __attribute__((address_space(1))) &) {
    return 1;
  }
  consteval int immediate(this const Record __attribute__((address_space(1))) &
                          self) {
    return self.value + self.array[1];
  }
  template <class Self>
  constexpr auto address(this Self& self) {
    return &self.value;
  }
};
using RecordOne = Record __attribute__((address_space(1)));
constexpr int Record::* data_member = &Record::value;
static_assert(__is_same(decltype(((RecordOne*)nullptr)->*data_member), int&));
static_assert(__is_same(decltype(((const RecordOne*)nullptr)->*data_member),
                        const int&));
constexpr Record ordinary{3, {4, 5}};
constexpr RecordOne annotated{7, {11, 13}};
constexpr RecordOne zero{};
static_assert(ordinary.select() == 0);
static_assert(annotated.select() == 1);
static_assert(annotated.immediate() == 20);
static_assert(annotated.value == 7);
static_assert(zero.value == 0);
constexpr auto member = &annotated.value;
static_assert(__is_same(decltype(member), const IntOne* const));
static_assert(*member == 7);
static_assert(__is_same(decltype(annotated.address()), const IntOne*));
static_assert(*annotated.address() == 7);

constexpr int read(const RecordOne& value) { return value.array[0]; }
consteval int evaluate(const RecordOne& value) { return value.value; }
static_assert(read(annotated) == 11);
static_assert(evaluate(annotated) == 7);

struct Immediate {
  consteval int value() {
    static_assert(__is_same(decltype(this), Immediate*));
    return 1;
  }
  consteval int value() const {
    static_assert(__is_same(decltype(this), const Immediate*));
    return 2;
  }
};
constexpr Immediate immediate{};
static_assert(Immediate{}.value() == 1);
static_assert(immediate.value() == 2);

struct Fallback {
  constexpr int select() const { return 1; }
  static constexpr int select(int = 0) { return 2; }
};
using FallbackOne = Fallback __attribute__((address_space(1)));
constexpr FallbackOne fallback{};
static_assert(fallback.select() == 2);

constexpr int plain = 17;
constexpr const IntOne& reference = (const IntOne&)plain;
static_assert(reference == 17);
