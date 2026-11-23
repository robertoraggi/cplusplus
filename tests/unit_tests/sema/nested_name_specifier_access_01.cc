// RUN: %cxx -std=c++26 -verify -fsyntax-only %s

class Any {
  template <class T>
  struct TypeId {
    static char id;
  };

 public:
  template <class T>
  auto isa() const -> bool {
    return &TypeId<T>::id != nullptr;
  }
};

template <class T>
char Any::TypeId<T>::id = 1;

bool isInt = Any{}.isa<int>();

template <class T>
struct Outer {
 private:
  struct Nested {
    struct Inner {
      static T value;
      auto get() -> T;
    };
  };

 public:
  auto sum() -> T { return Nested::Inner::value + typename Nested::Inner{}.get(); }
};

template <class T>
T Outer<T>::Nested::Inner::value = 2;

template <class T>
auto Outer<T>::Nested::Inner::get() -> T {
  return 3;
}

int outerSum = Outer<int>{}.sum();

struct Target {
  static int k;
};

class Hides {
  using Alias = Target;
};

int Hides::Alias::k = 1;  // expected-error {{'Alias' is a private member of 'Hides'}}

template <class...>
using void_t = void;

template <class D, class = void>
struct Is {
  static constexpr bool value = false;
};

template <class D>
struct Is<D, void_t<typename D::A::B>> {
  static constexpr bool value = true;
};

struct Public {
  struct A {
    using B = int;
  };
};

class Private {
  struct A {
    using B = int;
  };
};

class Befriends {
  struct A {
    using B = int;
  };

  template <class, class>
  friend struct Is;
};

static_assert(Is<Public>::value);
static_assert(!Is<Private>::value);
static_assert(Is<Befriends>::value);

template <class D>
concept Deep = requires { typename D::A::B; };

static_assert(Deep<Public>);
static_assert(!Deep<Private>);

template <class D>
constexpr auto pick(D) -> typename D::A::B {
  return 1;
}

constexpr auto pick(...) -> int { return 2; }

static_assert(pick(Public{}) == 1);
static_assert(pick(Private{}) == 2);

class Grants {
  struct A {
    struct B {
      static constexpr int v = 1;
    };
  };

  template <class T>
  friend constexpr auto useFriend(T) -> int;
};

template <class T>
constexpr auto useFriend(T) -> int {
  return T::A::B::v;
}

static_assert(useFriend(Grants{}) == 1);

template <class T>
constexpr auto useStranger(T) -> int {
  return T::A::B::v;  // expected-error {{'A' is a private member of 'Grants'}}
}

int stranger = useStranger(Grants{});  // expected-note {{in instantiation of function template specialization 'useStranger<Grants>' requested here}}
