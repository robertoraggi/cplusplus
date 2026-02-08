// RUN: %cxx -verify -fcheck %s

struct Pair {
  int first;
  int second;

  constexpr Pair(int a, int b) : first(a), second(b) {}
  constexpr int sum() { return first + second; }
  constexpr int product() { return first * second; }
};

constexpr Pair p(3, 7);

constexpr int s = p.sum();
static_assert(s == 10, "Expected sum to be 10");

constexpr int pr = p.product();
static_assert(pr == 21, "Expected product to be 21");

constexpr int pf = p.first;
static_assert(pf == 3, "Expected first to be 3");

constexpr int ps = p.second;
static_assert(ps == 7, "Expected second to be 7");

struct Point {
  int x;
  int y;

  constexpr Point(int ax, int ay) : x(ax), y(ay) {}
  constexpr int manhattan() { return x + y; }
};

constexpr Point pt(4, 5);
constexpr int m = pt.manhattan();
static_assert(m == 9, "Expected Manhattan distance to be 9");

constexpr int px = pt.x;
static_assert(px == 4, "Expected x to be 4");

constexpr int py = pt.y;
static_assert(py == 5, "Expected y to be 5");

constexpr int compute_sum(int a, int b) {
  Pair p(a, b);
  return p.sum();
}
constexpr int cs = compute_sum(10, 20);
static_assert(cs == 30, "Expected compute_sum(10, 20) to be 30");

constexpr Pair q(cs, 5);

constexpr int qs = q.sum();
static_assert(qs == 35, "Expected sum to be 35");
