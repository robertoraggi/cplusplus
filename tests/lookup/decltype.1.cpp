
namespace ns {
struct base {
  int a;
};
} // ::ns

ns::base f;

struct derived: decltype(f) {
  void test1() { a = 0; }
  void test2();
  decltype(f) x;
  decltype(nullptr) y;
};

void derived::test2() {
  a = 0;
}
