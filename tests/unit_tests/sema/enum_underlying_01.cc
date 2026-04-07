// RUN: %cxx -verify %s

enum Plain { a = 0, b = 1, c = 5 };
enum PlainNeg { x = -1, y = 0 };
enum class Scoped { red, green };

static_assert(__is_same(__underlying_type(Plain), unsigned int));
static_assert(__is_same(__underlying_type(PlainNeg), int));

auto test_scoped(Scoped s) -> bool {
  auto plain = Plain(0);
  return s == Scoped::red || s != Scoped::green;
}
