
struct foo {
  int x;
  int* p;
  auto get_x() -> int { return x; }
  auto get_x_again() -> decltype(x);
  auto get_p() -> decltype(p);
  auto get_this() -> decltype(this);
  auto get_string() -> decltype("");
  auto get_null() -> decltype(nullptr);
};
