
struct test {
  struct iterator {
  };
  int method(iterator it);
};

int test::method(iterator it) {
  return 123;
}

