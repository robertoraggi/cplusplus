// RUN: %cxx -verify -fsyntax-only %s
// expected-no-diagnostics

struct Sum {
  int v;
  Sum& operator+=(int x) {
    v = v + x;
    return *this;
  }
};

template <class T>
struct View {};

template <class U>
struct TemplateSum {
  template <class T>
  TemplateSum& operator+=(const T&) {
    return *this;
  }
};

int test_compound_assignment() {
  Sum s{3};
  s += 4;
  TemplateSum<int> templateSum;
  templateSum += View<int>{};
  return s.v;
}

template <class T>
struct Ordered {
  int operator<=>(int) const;
};

bool reversed_rewritten_candidate() { return 1 < Ordered<int>{}; }
