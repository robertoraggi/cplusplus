// RUN: %cxx -verify -fsyntax-only %s

// expected-no-diagnostics

template <class C>
concept Any = true;
template <class C>
concept Small = sizeof(C) <= 4;

void constrained_pack(Any auto...);
void constrained_single(Any auto);
void constrained_mixed(Any auto, Small auto...);
void plain_pack(auto...);

template <class... Ts>
constexpr int count(Ts...) {
  return sizeof...(Ts);
}

void use() {
  constrained_pack(1, 2);
  constrained_pack();
  constrained_single(1);
  constrained_mixed(1, 'c', 2);
  plain_pack(1, 2);
}

auto lambda_constrained = [](Small auto... xs) { return sizeof...(xs); };
static_assert(lambda_constrained(1, 'c', 2) == 3);
