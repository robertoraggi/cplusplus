
struct base {
  int a;

  void clear() {
    a = 0;
  }
};

struct derived : base {
  void clear() {
    b = 0;
    a = 0;
  }

  int b;
};
