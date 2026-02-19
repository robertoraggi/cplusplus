// clang-format off
// RUN: %cxx -verify -fcheck %s

// expected-no-diagnostics

// Direct use of __type_pack_element
using A = __type_pack_element<0, int, double, char>;
using B = __type_pack_element<1, int, double, char>;
using C = __type_pack_element<2, int, double, char>;

static_assert(__is_same(A, int), "index 0");
static_assert(__is_same(B, double), "index 1");
static_assert(__is_same(C, char), "index 2");

// Use through a template alias (dependent case)
template <unsigned long _Ip, class... _Types>
using type_pack_element_t = __type_pack_element<_Ip, _Types...>;

static_assert(__is_same(type_pack_element_t<0, int, double>, int), "alias index 0");
static_assert(__is_same(type_pack_element_t<1, int, double>, double), "alias index 1");

// Single element pack
static_assert(__is_same(__type_pack_element<0, float>, float), "single element");
