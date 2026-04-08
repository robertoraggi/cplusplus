// RUN: %cxx -verify %s

// Variable template using noexcept as constexpr bool
template <typename T>
inline constexpr bool is_nothrow_default_constructible_v = noexcept(T());

struct NoThrow {
  NoThrow() noexcept;
};
struct Throws {
  Throws();
};

static_assert(is_nothrow_default_constructible_v<NoThrow>);
static_assert(!is_nothrow_default_constructible_v<Throws>);
