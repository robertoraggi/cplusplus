
struct test {
  struct iterator {
  };
  int method(iterator it);
  int x;
};

int test::method(iterator it) {
  x = x + 1;
  return x;
}

