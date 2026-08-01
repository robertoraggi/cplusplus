extern "C" {
auto printf(const char* format, ...) -> int;
}

template <typename... Args>
constexpr auto sum(Args... args) -> int {
  return (... + args);
}

template <typename... Args>
auto print_all(Args... args) -> void {
  (..., printf("Arg: %d\n", args));
}

auto main() -> int {
  constexpr int total = sum(1, 2, 3, 4, 5);
  printf("Constexpr Sum (1..5): %d\n", total);

  print_all(10, 20, 30);
  return 0;
}
