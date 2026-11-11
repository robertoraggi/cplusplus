// RUN: %cxx -std=c++20 -verify -fsyntax-only %s
// expected-no-diagnostics

struct R {
  int value;

  template <int I>
  constexpr int get() const {
    return value + I;
  }
};

struct Opt {
  int v = 0;
};

struct NestedOpt {
  Opt opt;
};

constexpr R f(int value, Opt opt) { return {value + opt.v}; }
constexpr R nested(NestedOpt opt) { return {opt.opt.v}; }

template <class T>
struct W {
  constexpr int eval(T) {
    auto r = f(1, {});
    return r.get<0>();
  }

  constexpr int next() { return 2; }
};

constexpr int use() {
  W<int> w;
  return w.eval(1) + w.next();
}

template <class T>
constexpr int nonempty(T) {
  auto r = f(1, {2});
  auto n = nested({{3}});
  auto pointer = &f;
  auto p = pointer(4, {});
  return r.get<0>() + n.get<0>() + p.get<0>();
}

constexpr R select(int value) { return {value}; }

struct Other {
  template <int I>
  constexpr int get() const {
    return -I;
  }
};

constexpr Other select(int*) { return {}; }

struct Pair {
  int first;
  int second;
};

constexpr Other select(Pair) { return {}; }

template <class T>
constexpr int dependent() {
  auto r = select({T{}});
  return r.template get<1>();
}

template <int... Ns>
constexpr int pack() {
  auto r = nested({{Ns...}});
  return r.template get<0>();
}

template <int... Ns>
constexpr int packOverload() {
  auto r = select({Ns...});
  return r.template get<1>();
}

constexpr int outside() {
  auto r = f(1, {});
  return r.get<0>();
}

template <class T>
constexpr int typed(T) {
  auto r = f(1, Opt{});
  return r.get<0>();
}

static_assert(use() == 3);
static_assert(nonempty(0) == 10);
static_assert(dependent<int>() == 1);
static_assert(dependent<int*>() == -1);
static_assert(pack<>() == 0);
static_assert(pack<5>() == 5);
static_assert(packOverload<5>() == 6);
static_assert(packOverload<5, 6>() == -1);
static_assert(outside() == 1);
static_assert(typed(0) == 1);
