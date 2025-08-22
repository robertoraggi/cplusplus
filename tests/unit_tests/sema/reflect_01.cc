// RUN: %cxx -verify -fcheck %s

namespace std {
namespace meta {
using info = __builtin_meta_info;
}
}  // namespace std

constexpr std::meta::info int_ty = ^^int;
static_assert(__is_same(decltype(int_ty), const std::meta::info));

constexpr std::meta::info ptr_ty = ^^const void*;
static_assert(__is_same(decltype(ptr_ty), const std::meta::info));

constexpr std::meta::info z = ^^123;
static_assert(__is_same(decltype(z), const std::meta::info));

constexpr int x = [:z:];
static_assert(x == 123);

constexpr[:int_ty:] i = 123;
static_assert(__is_same(decltype(i), const int));

static_assert(i == 123);

constexpr[:ptr_ty:] ptr = nullptr;
static_assert(__is_same(decltype(ptr), const void* const));
