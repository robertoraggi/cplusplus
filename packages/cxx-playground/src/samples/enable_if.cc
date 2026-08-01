extern "C" {
auto printf(const char* format, ...) -> int;
}

namespace std {
template <bool B, typename T = void>
struct enable_if {};

template <typename T>
struct enable_if<true, T> {
  using type = T;
};
}  // namespace std

template <typename T, typename std::enable_if<sizeof(T) <= 4>::type* = nullptr>
auto process(T val) -> void {
  printf("Small 32-bit type (size %d bytes): %d\n", static_cast<int>(sizeof(T)),
         static_cast<int>(val));
}

template <typename T, typename std::enable_if<(sizeof(T) > 4)>::type* = nullptr>
auto process(T val) -> void {
  printf("Large type (size %d bytes)\n", static_cast<int>(sizeof(T)));
}

auto main() -> int {
  int a = 42;
  double b = 3.14159;

  process(a);
  process(b);
  return 0;
}
