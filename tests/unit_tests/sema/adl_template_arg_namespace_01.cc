// RUN: %cxx -verify -fcheck %s

namespace lib {

template <typename T>
struct Box {};

}  // namespace lib

namespace user {

struct Token {};

constexpr auto call(lib::Box<Token>) -> int { return 7; }

}  // namespace user

auto g() -> int {
  lib::Box<user::Token> value{};
  return call(value);
}
