// RUN: %cxx -verify -fsyntax-only -std=c++26 %s

template <class T>
concept Complete = sizeof(T) > 0;

template <class T>
struct Alloc {};

template <class T>
struct Other {};

template <template <class> class A>
struct Impl {};

template <template <class> class A>
using PC = Impl<A>;
template <template <class> class A>
using PC = Impl<A>;
PC<Alloc> pc;

template <template <class> class A>
constexpr int declaredThenDefined(Impl<A>);
template <template <class> class B>
constexpr int declaredThenDefined(Impl<B>) {
  return 1;
}
static_assert(declaredThenDefined(Impl<Alloc>{}) == 1);

template <template <class> class A, template <class> class B>
constexpr int renamed(Impl<A>, Impl<B>);
template <template <class> class B, template <class> class A>
constexpr int renamed(Impl<B>, Impl<A>) {
  return 2;
}
static_assert(renamed(Impl<Alloc>{}, Impl<Other>{}) == 2);

template <template <class> class A, template <class> class B>
constexpr int swapped(Impl<A>, Impl<B>) {
  return 1;
}
template <template <class> class A, template <class> class B>
constexpr int swapped(Impl<B>, Impl<A>) {
  return 2;
}
static_assert(swapped<Alloc, Other>(Impl<Alloc>{}, Impl<Other>{}) == 1);
static_assert(swapped<Alloc, Other>(Impl<Other>{}, Impl<Alloc>{}) == 2);

template <template <Complete> class A>
constexpr int constrainedHead(Impl<A>) {  // expected-note {{candidate function}}
  return 1;
}
template <template <class> class A>
constexpr int constrainedHead(Impl<A>) {  // expected-note {{candidate function}}
  return 2;
}
int ambiguous = constrainedHead(Impl<Alloc>{});  // expected-error {{call to function is ambiguous}}
