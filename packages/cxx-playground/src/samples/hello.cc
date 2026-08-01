extern "C" {
auto printf(const char* format, ...) -> int;
}

auto main() -> int {
  printf("Hello, World!\n");
  return 0;
}
